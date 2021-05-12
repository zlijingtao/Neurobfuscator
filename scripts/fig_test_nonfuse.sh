#!/bin/bash
#Use to generate fuse/non-fuse timeline figure.
cd ../seq_obfuscator

python torch_relay_obfuscate_nvprof.py

python torch_relay_obfuscate_nvprof.py --non_fuse

#When finished, open a new terminal and type: "nvvp &" to open nvidia visual profiler. Open generated sql files in the GUI.
