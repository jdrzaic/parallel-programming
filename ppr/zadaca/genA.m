#!/usr/bin/octave -qf

if (nargin != 3)
    usage([program_name() " M N outfile"]);
endif

m = base2dec(argv(){1}, 10);
n = base2dec(argv(){2}, 10);
outFileName = argv(){3};

A = randn(m, n);

d = abs(diag(A));

max(max(abs(A) - diag(d), [], 2) + d)

outFile = fopen(outFileName, "wb");
fwrite(outFile, A, "double");
fclose(outFile);

