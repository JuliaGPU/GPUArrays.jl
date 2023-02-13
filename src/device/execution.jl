# kernel execution

# how many threads and blocks `kernel` needs to be launched with, passing arguments `args`,
# to fully saturate the GPU. `elements` indicates the number of elements that needs to be
# processed, while `elements_per_threads` indicates the number of elements this kernel can
# process (i.e. if it's a grid-stride kernel, or 1 if otherwise).
#
# this heuristic should be specialized for the back-end, ideally using an API for maximizing
# the occupancy of the launch configuration (like CUDA's occupancy API).
function launch_heuristic(backend::B, kernel, args...;
                          elements::Int,
                          elements_per_thread::Int) where B <: Backend
    return (threads=256, blocks=32)
end

# determine how many threads and blocks to actually launch given upper limits.
# returns a tuple of blocks, threads, and elements_per_thread (which is always 1
# unless specified that the kernel can handle a number of elements per thread)
function launch_configuration(backend::B, heuristic;
                              elements::Int,
                              elements_per_thread::Int) where B <: Backend
    threads = clamp(elements, 1, heuristic.threads)
    blocks = max(cld(elements, threads), 1)

    if elements_per_thread > 1 && blocks > heuristic.blocks
        # we want to launch more blocks than required, so prefer a grid-stride loop instead
        ## try to stick to the number of blocks that the heuristic suggested
        blocks = heuristic.blocks
        nelem = cld(elements, blocks*threads)
        ## only bump the number of blocks if we really need to
        if nelem > elements_per_thread
            nelem = elements_per_thread
            blocks = cld(elements, nelem*threads)
        end
        (; threads, blocks, elements_per_thread=nelem)
    else
        (; threads, blocks, elements_per_thread=1)
    end
end
