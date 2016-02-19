#!/usr/bin/octave -qf

if (nargin != 2)
    usage([program_name() " M N outfile"]);
endif

m = base2dec(argv(){1}, 10);

fileA = fopen(argv(){2}, "rb");
A = fread(fileA, [m, 1], "double")'

A

