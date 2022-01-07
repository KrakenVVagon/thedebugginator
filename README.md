# thedebugginator
thedebugginator is an event degredation detector that uses an autoencoder to detect anomlies inside telemetry events and 

SETUP:
Contains Dockerfile as well as build/run scripts for dockerization. Run scripts from parent directory.
requirements.txt is a text file which contains the versions of Python modules to install via pip.
notebook.sh is a bash file (to use inside the container) that will run a jupyter notebook for dev work. The notebook will be ported to 127.0.0.1:7745 by default.

DATASETS:
Contains training data for autoencoder. Two 1k anomaly files and one 10k proper files.