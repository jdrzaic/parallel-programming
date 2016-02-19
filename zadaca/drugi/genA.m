#!/usr/bin/octave -qf

if (nargin != 3)
    usage([program_name() " M N outfile"]);
endif

m = base2dec(argv(){1}, 10);
n = base2dec(argv(){2}, 10);
outFileName = argv(){3};

A = rand(m, n);

for i = 1:n
	A(:,i) /= norm(A(:,i));
end

A'*A

disp(A(1:min(10, m), 1:min(10, n)));

outFile = fopen(outFileName, "wb");
fwrite(outFile, A', "double");
fclose(outFile);

