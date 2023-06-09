# Quick start

## Generate some basic data
python 01_generate_data.py

## Load and visualize the first session (png and gif output)
python 90_session_plotter.py --session ./sessions-default/session_00000000.pkl 

## Train a model
python 12_task2_model_training.py --dataset ./sessions-default

## Generate more data
bash 00_setup.sh
