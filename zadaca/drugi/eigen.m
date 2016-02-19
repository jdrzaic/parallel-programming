#!/usr/bin/octave -qf

if (nargin != 3)
    usage([program_name() " M N A.dat"]);
endif

m = base2dec(argv(){1}, 10);
n = base2dec(argv(){2}, 10);

fileA = fopen(argv(){3}, "rb");
A = fread(fileA, [n, m], "double")';

fclose(fileA);

s = eig(A'*A);
disp(2 / (s(n) + s(1)))
