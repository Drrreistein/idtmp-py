#!/bin/bash

cat $1.log | grep true_feasible >> $1.txt
cat $1.log | grep true_infeasible >> $1.txt
cat $1.log | grep false_infeasible >> $1.txt
cat $1.log | grep false_feasible >> $1.txt