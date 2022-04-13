# thedebugginator

thedebugginator is a module to create deep learning models for anomaly detection.

Currently supports:
- autoencoders

## Project tree

```
+---data
|   +---interim
|       \ partially processed datasets that have gone through only some of the preprocessing steps
|   +---processed
|       \ fully processed datasets that can be directly input into a model
|   +---raw
|       \ raw extracts from the database
+---debugginator
|   \ working module.
|   +---data
|   +---edo
|   +---models
|   +---visualization
+---models
|   \ for trained and saved models to be used by run_debugginator.py or used for testing
+---notebooks
|   \ save jupyter notebooks here
+---reports
|   +---figures
+---scripts
    +---run
    +---setup
    +---examples
	\ example scripts showing training, testing, running and saving of models and data
```

## Setup and Installation

thedebugginator comes with a Dockerfile and scripts to build, run and start the associated image as well as a setup.py for installation

### Setup

On windows the following can be used to access the docker dev image

```bash
# builds the docker image
%cd%/scripts\setup\dockerBuild.bat

# creates the docker container. also will start and enter the container
%cd%/scripts\setup\dockerRun.bat

# starts the docker container if it is not currently running
%cd%/scripts\run\dockerStart.bat

# enters the docker container once it is already running
%cd%/scripts\run\dockerExec.bat
```

Bash scripts for linux distros will be created later. Docker command lines should be used

### Installation

Pip can be used to install thedebugginator from the main directory

```bash
pip install .
```

## Usage

Please see the folders under scripts/examples for some example scripts and use-case scenarios.

To use a notebook version run

```bash
/root/thedebugginator/scripts/run/notebook.sh
```

This will open a Jupyter notebook instance which is **locally** accessible from 127.0.0.1:7745