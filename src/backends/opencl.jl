module CLBackend
using ..GPUArrays
using OpenCL
const cl = OpenCL
import GPUArrays: buffer, create_buffer, Context, GPUArray

immutable CLContext <: Context
	device 
	context
	queue
	kernel_dict
	function CLContext(;device_type=:gpu)
		device = first(cl.devices(device_type))
		ctx    = cl.Context(device)
		queue  = cl.CmdQueue(ctx)
		new(
			device, ctx, queue,
			Dict()
		)
	end
end

function init(;kw_args...)
	global compute_context = CLContext(;kw_args...)
end

typealias CLArray{T, N} GPUArray{cl.Buffer{T}, T, N, CLContext}

# Constructor
function (::Type{Array}){T, N}(A::CLArray{T, N})
	Array{T,N}(A)
end

function (::Type{Array{T, N}}){T, N}(A::CLArray{T, N})
	hA = similar(Array{T, N}, size(A))
	copy!(A.context.queue, hA, buffer(A))
	hA
end

function create_buffer{T, N}(ctx::CLContext, A::AbstractArray{T, N}, flag = :rw)
	cl.Buffer(T, ctx.context, (flag, :copy), hostbuf=A)
end
function CLArray(A::AbstractArray, flag = :rw)
	ctx = compute_context::CLContext
	buf = create_buffer(ctx, A, flag)
	GPUArray(buf, size(A), ctx)
end

function Base.similar{T, N}(::Type{CLArray{T, N}}, sz::Tuple, flag = :rw)
	ctx = compute_context::CLContext
	buf = cl.Buffer(T, ctx.context, flag, prod(sz))
	GPUArray(buf, sz, ctx)
end

const cl_type_map = Dict(
	Float32 => "float",
	Complex{Float32} => "c_float_t",
	Int32 => "int"
)

const cl_fun_map = Dict{Type, String}(
)

function type_expr(typ, changes=false)
	(cltype_prefix(typ, changes) *
	cltype_name(typ) *
	cltype_postfix(typ))
end
cltype_prefix(typ, changes=false) = changes ? "" : "const "
function cltype_prefix{T, N}(::Type{CLArray{T, N}}, changes=false)
	isconst = !changes # || is_readonly(typ)
	const_str = isconst ? "const " : ""
	"__global $const_str"
end

function cltype_name(typ)
	T = typ <: GPUArray ? eltype(typ) : typ
	get(cl_type_map, T) do
		error("Type $typ not supported by OpenCL backend")
	end
end
cltype_postfix{T, N}(::Type{CLArray{T, N}}) = "* "
cltype_postfix(typ) = ""

function getindex_expr{T, N}(typ::Type{CLArray{T, N}}, sym, idx = :i)
	"$sym[$idx]"
end
function getindex_expr(typ, sym, idx="i")
	"$sym"
end
function setindex_expr{T, N}(typ::Type{CLArray{T, N}}, sym, value, idx="i")
	"$sym[$idx] = $value"
end


function is_infix(x)
	isa(+, x) ||
	isa(*, x) ||
	isa(-, x) ||
	isa(/, x) ||
	isa(==, x)
end

function apply_expr(op, a, b)
	fun = replace(string(op.name.name), "#", "") # LOL
	if is_infix(op)
		return "$a $fun $b"
	else
		return "$fun($a, $b)"
	end
end

# 16x vector loads, for now (still experimenting)
const GRID = 16

# Our little silly JIT!
function map_kernel{T1, T2, T3}(op, out::T1, A::T2, B::T3)
	ctx = compute_context::CLContext
	kernel, source = get!(ctx.kernel_dict, (op, T1, T2, T3)) do
		ai, bi = getindex_expr(A, :A, :i), getindex_expr(B, :B, :i)
		outi = setindex_expr(out, :out, :value, :i)
		T = type_expr(eltype(out), true)
		source = """
		__kernel void map_kernel(
		        $(type_expr(out, true)) out,
		        $(type_expr(A)) A,
		        $(type_expr(B)) B
		    ){
			int i = get_global_id(0);
			$(T)$(GRID) value1a = vload$(GRID)(i, A);
			$(T)$(GRID) value1b = vload$(GRID)(i, B);
			vstore$(GRID)($(apply_expr(op, "value1a", "value1b")), i, out);
		}
		"""
		# println(source)
		program = cl.Program(ctx.context, source=source)
	  cl.build!(program)
		cl.Kernel(program, "map_kernel"), source
	end
	kernel
end

function workgroup_heuristic(::typeof(broadcast!), out, A, B)
	Csize_t[div(length(out), GRID)], C_NULL
end

function Base.broadcast{T1, T2, N}(op, A::CLArray{T1, N}, B::CLArray{T2, N})
	T = Base.promote_op(op, T1, T2)
	S = size(A)
	out = similar(CLArray{T, N}, S, :w)
	broadcast!(op, out, A, B)
	out
end

@generated function Base.broadcast!{T1, T2, T3, N}(
		op, out::CLArray{T1, N}, A::CLArray{T2, N}, B::CLArray{T3, N}
	)
	kernel = map_kernel(op, out, A, B)
	ret_event = Array(cl.CL_event, 1)
	q = compute_context.queue::cl.CmdQueue
	quote
		# moved this code out of OpenCL.jl to remove performances hurdles
		cl.set_args!($(kernel), buffer(out), buffer(A), buffer(B))
		gsize, lsize = workgroup_heuristic(broadcast!, out, A, B)
	    goffset = C_NULL
	    n_events = cl.cl_uint(0)
	    wait_event_ids = C_NULL
	    cl.api.clEnqueueNDRangeKernel(
			$(q.id), $(kernel.id), cl.cl_uint(1), goffset, gsize, lsize,
	    	n_events, wait_event_ids, $ret_event
		)
	    cl.finish($(q))
		nothing
	end
end

end #CLBackend



#     __global  *out, KParam oInfo,
#     uint groups_0, uint groups_1, uint num_odims)
#
#     {
#
#
#         uint groupId  = get_group_id(1) * get_num_groups(0) + get_group_id(0)
#         uint threadId = get_local_id(0)
#         int idx = groupId * get_local_size(0) * get_local_size(1) + threadId
#         if (idx >= oInfo.dims[3] * oInfo.strides[3]) return;
#
#
#      out[idx] = val
#
# }
# source = """
# __kernel void map_kernel(
# 				$(type_expr(out, true)) out,
# 				$(type_expr(A)) A,
# 				$(type_expr(B)) B
# 		){
# 		int i = get_global_id(0);
# 			int li = get_local_id(0);
# 		__local $T tmpA[128];
# 		__local $T tmpB[128];
# 		__local $T tmpout[128];
# 		async_work_group_copy(tmpA, A+i-li, 128, 0);
# 		async_work_group_copy(tmpB, B+i-li, 128, 0);
# 		async_work_group_copy(tmpout, out+i-li, 128, 0);
# 		for(int k=0; k<128, k++){
# 			tmpout[k] = $(apply_expr(op, "tmpA[k]", "tmpB[k]"));
# 		}
# 		async_work_group_copy(out+i-li, tmpout, 128, 0);
# 		$outi;
# }
# """