import numpy.random as nprnd
import sys
filename = sys.argv[2]
f = open(filename, 'w')
v = sys.argv[1]
a = nprnd.randint(2, size = int(v))
f.write("%s\n" % v)
f.write("%s\n" % sum(a))
for item in a:
	f.write("%s\n" % item)

