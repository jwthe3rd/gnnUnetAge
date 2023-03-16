#!/bin/bash

for d in */; do
	
	cp -i ../data/baffle8/l10_${d%/}.pt ./${d}/
	echo ${d}

done
