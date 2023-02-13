#!/bin/bash

cd tests
for d in */; do
	echo ${d}
	python ../extract_feats.py -direct "${d%/}"
done
	
