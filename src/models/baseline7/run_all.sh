#!/bin/bash

ROOT='/home/rafael/Dataframe'

declare -a ds=('esic' 'enblog' 'pan13')
declare -a t=('gender' 'age')
# 'education' 'religion' 'profession' 'region' 'city' 'politics' 'it')

for i in "${ds[@]}"
do
	#for j in "${t[@]}"
	#do
	python3 baseline3_new.py $ROOT $i
	#done
done


