function thread_blocks_heuristic(len::Integer)
    threads = min(len, 256)
    blocks = ceil(Int, len/threads)
    blocks = blocks * threads
    blocks, threads
end
