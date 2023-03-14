#!/bin/bash
./run.sh baffle8
./inf.sh inf_config
cd tests
./gen_vtk.sh
