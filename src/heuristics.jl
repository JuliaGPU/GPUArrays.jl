function thread_blocks_heuristic(len::Integer)
    # TODO better threads default
    threads = min(len, 256)
    blocks = ceil(Int, len / threads)
    (blocks,), (threads,)
end
