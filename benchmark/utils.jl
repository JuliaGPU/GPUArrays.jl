macro af_bench(expr)
  begin
    r = $expr
    ArrayFire.eval(r)
    sync()
  end
end

macro cl_bench(expr)
  begin
    r = $expr
    ArrayFire.eval(r)
    sync()
  end
end
