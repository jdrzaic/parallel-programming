#!/bin/bash

gcc -o $1 $1.c -std=gnu99 -pthread -lblas &&

./genP.py $3 $4 $5 &&

time ./$1 $2 $3 $4 $5 > res.txt &&

./checkP.py res.txt $5 $3 $4
