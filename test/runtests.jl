using GPUArrays, Test, Pkg


@testset "Enzyme JLArray: TypeTree" begin
    
    using Enzyme
    using JLArrays
    using LLVM

    import Enzyme: typetree, TypeTree, API, make_zero

    ctx = LLVM.Context()
    dl = string(LLVM.DataLayout(LLVM.JITTargetMachine()))

    tt(T) = string(typetree(T, ctx, dl))
    @test tt(JLArray{Float64, 1}) == "{[0]:Pointer, [0,0]:Pointer, [0,0,-1]:Pointer, [0,0,0,0]:Pointer, [0,0,0,0,-1]:Float@double, [0,0,0,8]:Integer, [0,0,0,9]:Integer, [0,0,0,10]:Integer, [0,0,0,11]:Integer, [0,0,0,12]:Integer, [0,0,0,13]:Integer, [0,0,0,14]:Integer, [0,0,0,15]:Integer, [0,0,0,16]:Integer, [0,0,0,17]:Integer, [0,0,0,18]:Integer, [0,0,0,19]:Integer, [0,0,0,20]:Integer, [0,0,0,21]:Integer, [0,0,0,22]:Integer, [0,0,0,23]:Integer, [0,0,0,24]:Integer, [0,0,0,25]:Integer, [0,0,0,26]:Integer, [0,0,0,27]:Integer, [0,0,0,28]:Integer, [0,0,0,29]:Integer, [0,0,0,30]:Integer, [0,0,0,31]:Integer, [0,0,0,32]:Integer, [0,0,0,33]:Integer, [0,0,0,34]:Integer, [0,0,0,35]:Integer, [0,0,0,36]:Integer, [0,0,0,37]:Integer, [0,0,0,38]:Integer, [0,0,0,39]:Integer, [0,0,16,-1]:Integer, [0,8]:Integer, [8]:Integer, [9]:Integer, [10]:Integer, [11]:Integer, [12]:Integer, [13]:Integer, [14]:Integer, [15]:Integer, [16]:Integer, [17]:Integer, [18]:Integer, [19]:Integer, [20]:Integer, [21]:Integer, [22]:Integer, [23]:Integer}"
end


include("testsuite.jl")


@testset "JLArray" begin
    using JLArrays

    jl([1])

    TestSuite.test(JLArray)
end

@testset "Array" begin
    TestSuite.test(Array)
end
