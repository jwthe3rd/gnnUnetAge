#!/bin/bash

cd data/baffle7/

for f in *; do
	
	if [ "${f:0:2}" == "f_" ]; then
		echo ${f}
		python ../../qs.py -file "${f}"
	fi

done
