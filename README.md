# thedebugginator
thedebugginator is an event degredation detector that uses an autoencoder to detect anomlies inside telemetry events.

scripts/setup:
Contains Dockerfile as well as build/run scripts for dockerization. Run scripts from home directory.
requirements.txt is a text file which contains the versions of Python modules to install via pip.
notebook.sh is a bash file (to use inside the container) that will run a jupyter notebook for dev work. The notebook will be ported to 127.0.0.1:7745 by default.

data/raw:
Contains raw data from the database (game_watchdogslegion_prod.fact_playerkill) before it is processed for the autoencoder.

data/interim:
Contains the training and test splits required for training and preprocessing the model

data/processed:
Contains an example of the final processed data that can be read by the autoencoder

lib:
Contains some modules useful for parts of thedebugginator's workflow

models:
Contains trained models that are used during the prediction scrpits

notebooks:
Jupyter notebooks used during development

reports:
Any reports created using thedebugginator go here

debugginator:
Project source code. setup.py turns this folder into the high level module and lets it be pip installable

debugginator/data:
Classes and functions used to extract data and process the data to be consumed by the autoencoding models

debugginator/model:
Classes and functions to train model and use model to make predictions

debugginator/visualization:
Classes and functions to visualize model outputs or training parameters