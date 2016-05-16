using StatsBase

function gen_matrix(m::Int, n::Int, k::Int, filename)
    A = rand(m, k) - 0.5;
    B = rand(k, n) - 0.5;
    R = A*B;
    open(filename, "w") do fp
        write(fp, R)
    end
    display(filename)
    display(R)
    R
end

function gen_test(m::Int, n::Int, p::Int, k1::Int, k2::Int, afile, bfile,
                  dfile)
    A = gen_matrix(m, n, k1, afile)
    B = gen_matrix(p, n, k2, bfile)
    open(dfile, "w") do fp
        print_joined(fp, ["-m", m, "-n", n, "-p", p, afile, bfile, "\n"], " ")
    end
end

