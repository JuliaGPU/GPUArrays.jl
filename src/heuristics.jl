function thread_blocks_heuristic(len::Integer)
    # TODO better threads default
    threads = clamp(len, 1, 256)
    blocks = max(ceil(Int, len / threads), 1)
    (blocks,), (threads,)
end
