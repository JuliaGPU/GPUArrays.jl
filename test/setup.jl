using Distributed, Test, JLArrays

include("testsuite.jl")

using Random

if VERSION >= v"1.13.0-DEV.1044"
using Base.ScopedValues
end

## entry point

function runtests(f, name)
    function inner()
        # generate a temporary module to execute the tests in
        mod_name = Symbol("Test", rand(1:100), "Main_", replace(name, '/' => '_'))
        mod = @eval(Main, module $mod_name end)
        @eval(mod, using Test, Random, JLArrays)

        let id = myid()
            wait(@spawnat 1 print_testworker_started(name, id))
        end

        ex = quote
            GC.gc(true)
            Random.seed!(1)
            JLArrays.allowscalar(false)

            @timed @testset $"$name" begin
                $f()
            end
        end
        data = @static if VERSION < v"1.13.0-DEV.1044"
            Core.eval(mod, ex)
        else
            @with Test.TESTSET_PRINT_ENABLE => false Core.eval(mod, ex)
        end
        #data[1] is the testset

        # process results
        cpu_rss = Sys.maxrss()
        if VERSION >= v"1.11.0-DEV.1529"
            tc = Test.get_test_counts(data[1])
            passes,fails,error,broken,c_passes,c_fails,c_errors,c_broken =
                tc.passes, tc.fails, tc.errors, tc.broken, tc.cumulative_passes,
                tc.cumulative_fails, tc.cumulative_errors, tc.cumulative_broken
        else
            passes,fails,errors,broken,c_passes,c_fails,c_errors,c_broken =
                Test.get_test_counts(data[1])
        end
        if data[1].anynonpass == false
            data = ((passes+c_passes,broken+c_broken),
                    data[2],
                    data[3],
                    data[4],
                    data[5])
        end
        res = vcat(collect(data), cpu_rss)

        GC.gc(true)
        res
    end

    @static if VERSION >= v"1.13.0-DEV.1044"
        @with Test.TESTSET_PRINT_ENABLE=>false begin
            inner()
        end
    else
        old_print_setting = Test.TESTSET_PRINT_ENABLE[]
        Test.TESTSET_PRINT_ENABLE[] = false
        try
            inner()
        finally
            Test.TESTSET_PRINT_ENABLE[] = old_print_setting
        end
    end
end

nothing # File is loaded via a remotecall to "include". Ensure it returns "nothing".
