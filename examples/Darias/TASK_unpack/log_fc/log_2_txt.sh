#!/bin/bash
cat $1.log | grep 'final_visits ' >> $1.txt 
cat $1.log | grep 'task_planning_time ' >> $1.txt 
cat $1.log | grep 'motion_refiner_time ' >> $1.txt 
cat $1.log | grep 'total_planning_time ' >> $1.txt 