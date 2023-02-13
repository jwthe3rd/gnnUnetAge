#!/bin/bash

cd tests

for d in */; do
	echo ${d}
	python ../extract_edges.py -direct "${d%/}"
done
	
