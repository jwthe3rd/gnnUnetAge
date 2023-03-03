#!/bin/bash

for f in */; do
	cd ${f}
	foamToVTK
	cd ..

done
