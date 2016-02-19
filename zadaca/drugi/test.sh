#!/bin/bash

gcc -o $1 $1.c -std=gnu99 -pthread -lblas &&

./genA.m $3 $4 A.dat &&

./genA.m $3 1 b.dat &&

./genA.m $4 1 x.dat &&

./$1 $2 $3 $4 $(./eigen.m $3 $4 A.dat) $5 A.dat b.dat x.dat xn.dat bin &&

./calcR.m $3 $4 A.dat b.dat xn.bin
