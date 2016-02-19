#!/usr/bin/python

import random
import sys
from array import *

args = sys.argv

m = int(args[1])
n = int(args[2])

f = open(args[3], 'w')
for i in range(m):
	f.write(''.join(random.choice('o.') for _ in range(n)))
	f.write('\n')
f.close()
