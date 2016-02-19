#!/usr/bin/octave -qf

if (nargin != 5)
    usage([program_name() " M N A.dat b.dat x.dat"]);
endif

m = base2dec(argv(){1}, 10);
n = base2dec(argv(){2}, 10);

fileA = fopen(argv(){3}, "rb");
A = fread(fileA, [n, m], "double")';
fileB = fopen(argv(){4}, "rb");
b = fread(fileB, [m, 1], "double");
fileX = fopen(argv(){5}, "rb");
x = fread(fileX, [n, 1], "double");

r = A*x - b;

A'*r



