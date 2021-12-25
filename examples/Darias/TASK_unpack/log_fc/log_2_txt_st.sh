#!/bin/bash
cat $1.log | grep 'true_in' >> $1.txt 
cat $1.log | grep 'true_feas' >> $1.txt 
cat $1.log | grep 'false_infeas' >> $1.txt 
cat $1.log | grep 'false_feas' >> $1.txt 