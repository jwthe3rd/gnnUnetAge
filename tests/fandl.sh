#!/bin/bash

cd extrap
for d in */; do
	echo ${d}
	if [ -f ./completed_${d%/}.txt ]; then
		echo ${d}
	else
		touch ./completed_${d%/}.txt
		python ../extract_feats.py -direct "${d%/}"
		python ../extract_labels.py -direct "${d}"
	fi
done
	
