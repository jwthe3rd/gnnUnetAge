#!/bin/bash
#
#
cd test_runs

for f in */; do

	if [ -f ../data/baffle8/f_${f%/}.pt ]; then

	mv ../data/baffle8/f_${f%/}.pt ${f}
	mv ../data/baffle8/l10_${f%/}.pt ${f}

	fi

done
