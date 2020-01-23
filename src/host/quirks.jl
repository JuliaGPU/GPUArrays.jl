# revert JuliaLang/julia#32867; avoid string interpolation
#
# NOTE: without contextual dispatch, we can only redefine methods where a GPU-specific
#       type occurs in the signature (or we'll get a "fatal precompilation failure" error)

if VERSION >= v"1.3.0-alpha.107"
    _bcs1(a::Integer, b::Integer) = a == 1 ? b : (b == 1 ? a : (a == b ? a : throw(DimensionMismatch("arrays could not be broadcast to a common size"))))
    _bcs1(a::Integer, b) = a == 1 ? b : (first(b) == 1 && last(b) == a ? b : throw(DimensionMismatch("arrays could not be broadcast to a common size")))
    _bcs1(a, b::Integer) = _bcs1(b, a)
    _bcs1(a, b) = Broadcast._bcsm(b, a) ? Broadcast.axistype(b, a) : (Broadcast._bcsm(a, b) ? Broadcast.axistype(a, b) : throw(DimensionMismatch("arrays could not be broadcast to a common size")))

    _bcs(::Tuple{}, ::Tuple{}) = ()
    _bcs(::Tuple{}, newshape::Tuple) = (newshape[1], _bcs((), Base.tail(newshape))...)
    _bcs(shape::Tuple, ::Tuple{}) = (shape[1], _bcs(Base.tail(shape), ())...)
    function _bcs(shape::Tuple, newshape::Tuple)
        return (_bcs1(shape[1], newshape[1]), _bcs(Base.tail(shape), Base.tail(newshape))...)
    end

    broadcast_shape(shape::Tuple) = shape
    broadcast_shape(shape::Tuple, shape1::Tuple, shapes::Tuple...) = broadcast_shape(_bcs(shape, shape1), shapes...)

    @inline combine_axes(A, B...) = broadcast_shape(axes(A), combine_axes(B...))
    combine_axes(A) = axes(A)

    Broadcast._axes(::Broadcasted{ArrayStyle{AT}}, axes::Tuple) where {AT <: GPUArray} = axes
    @inline Broadcast._axes(bc::Broadcasted{ArrayStyle{AT}}, ::Nothing) where {AT <: GPUArray} = combine_axes(bc.args...)
end
