#!/bin/bash
./winrun.sh baffle8
./wininf.sh inf_config
cd tests
./gen_vtk.sh
