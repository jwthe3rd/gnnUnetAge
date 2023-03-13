#!/bin/bash


for d in */; do
	echo ${d}
	if [ -f ../prepData2/completed_${d%/}.txt ]; then
		echo ${d}
	else
		touch ../prepData2/completed_${d%/}.txt
		# python extract_feats.py -direct "${d%/}"
		python extract_labels.py -direct "${d}"
	fi
done
	
