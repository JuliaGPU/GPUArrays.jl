# kernel execution

# TODO:
# - Rename KA device to backend
# - Who owns `AbstractGPUBackend`?
#   a; KernelAbstractions
#   b; GPUArraysCore
backend(a) = KernelAbstractions.get_device(a)