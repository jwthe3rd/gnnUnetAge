#!/bin/bash

cd tests
for d in */; do
	echo ${d}
	python ../extract_labels.py -direct "${d}"
done
	
