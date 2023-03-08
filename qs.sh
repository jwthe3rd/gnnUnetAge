#!/bin/bash

cd tests

for d in */; do
	cd ${d}
	for f in *; do
	if [ "${f:0:2}" == "f_" ]; then
		echo ${f}
		python ../../qs.py -file "${f}"
	fi
	done
	cd ..
done
