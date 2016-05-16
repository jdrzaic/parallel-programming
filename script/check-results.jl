function read_matrix(m::Int, n::Int, filename)
    open(fp -> read(fp, Float64, m, n), filename)
end


function check(m::Int, n::Int, p::Int, aorig, borig, afile, bfile, ufile,
               vfile, qfile)
    A = read_matrix(m, n, aorig)
    B = read_matrix(p, n, borig)
    Ap = triu(read_matrix(m, n, afile))
    Bp = triu(read_matrix(p, n, bfile))
    U = read_matrix(m, m, ufile)
    V = read_matrix(p, p, vfile)
    Q = read_matrix(n, n, qfile)
    display(Ap)
    display(Bp)
    [norm(U'*A*Q - Ap) norm(V'*B*Q - Bp)]
end

