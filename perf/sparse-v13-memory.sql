-- Nsight Systems sqlite export, device allocations in each pool-purged NVTX range.
-- Incremental live memory includes the result and is relative to the range start.
WITH ranges AS (
  SELECT n.start AS lo,n.end AS hi,coalesce(n.text,s.value) AS label
  FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
  WHERE coalesce(n.text,s.value) LIKE 'memory:%'
), events AS (
  SELECT r.label,m.start,m.rowid AS ord,
    CASE WHEN m.memoryOperationType=0 THEN m.bytes ELSE -m.bytes END AS delta,
    CASE WHEN m.memoryOperationType=0 THEN m.bytes ELSE 0 END AS allocated
  FROM ranges r JOIN CUDA_GPU_MEMORY_USAGE_EVENTS m ON m.start BETWEEN r.lo AND r.hi
  WHERE m.memKind=2
), live AS (
  SELECT *,sum(delta) OVER(PARTITION BY label ORDER BY start,ord ROWS UNBOUNDED PRECEDING)
    AS current FROM events
)
SELECT label,sum(allocated) AS device_allocated,max(current) AS peak_increment,
       sum(delta) AS end_increment
FROM live GROUP BY label;
