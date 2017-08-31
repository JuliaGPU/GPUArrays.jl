import Base: *, plan_ifft!, plan_fft!, plan_fft, plan_ifft, size, plan_bfft, plan_bfft!

for f in (:plan_fft, :plan_fft!)
    @eval function $f(A::JLArray; kw_args...)
        $f(buffer(A); kw_args...)
    end
end
for f in (:plan_bfft, :plan_bfft!)
    @eval function $f(A::JLArray, region; kw_args...)
        $f(buffer(A), region; kw_args...)
    end
end

function *(
        plan::Base.DFT.FFTW.cFFTWPlan{T, Direction, true, N},
        A::JLArray{T, N}
    ) where {T, N, Direction}
    JLArray(plan * buffer(A))
end
function *(
        plan::Base.DFT.FFTW.cFFTWPlan{T, Direction, false, N},
        A::JLArray{T, N}
    ) where {T, N, Direction}
    JLArray(plan * buffer(A))
end
