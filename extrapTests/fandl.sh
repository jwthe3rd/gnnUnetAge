#!/bin/bash


for d in */; do
	echo ${d}
#		touch ../prepData2/completed_${d%/}.txt
	# python extract_feats.py -direct "${d%/}"
	echo ${d}
	python extract_labels.py -direct "${d}"
	python extract_feats.py -direct "${d%/}"
done
	
