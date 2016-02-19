#!/usr/bin/python

import sys
import operator

a, file1, file2, m, n = sys.argv
m = int(m)
n = int(n)
f1 = open(file1, 'r')
f2 = open(file2, 'r')

all1 = f1.readlines()
tocheck = f2.read()
#print tocheck
same = 0
tail = 0
all2 = []
for i in range(m * (n + 1)):
	if tocheck[i] == 'o':
		all2.append("("+ str(i / (n + 1)) + ", " + str(i % (n + 1)) + ")\n")
#print "all1: " 
#print sorted(all1)
#print "all2: " 
#print all2
if all(map(operator.eq, sorted(all1), sorted(all2))):
	print "ok"
else:
	print "nije ok"
