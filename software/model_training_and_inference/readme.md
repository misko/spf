# Quick start

## Generate a reference dataset 

This creates a dataset with,
- 2 elements in a linear array attached to a moving (8m/s) detector
- 1 RF source (target) source moving around at 5m/s
- All motion is type 'bounce', when the detector or RF source hits the boundary it bounces
- 0.1 random white noise added to signal
- 512 timesteps per session
- Sampling from the radio 10Mhz, and doing this in burts every 0.3s
- RF source is fixed in center (x) and upper (y)

```01_generate_data.py --array-type linear --elements 2 --sources 1 --width 128 --detector-trajectory bounce --detector-speed 5 --sigma 0.1 --time-steps 512 --time-interval 0.3 --live true --output sessions-reference --sessions 200000000 --seed 1337 --reference true --source-speed 5 --detector-speed 8```

This should generate a data configuration, but not the dataset, since `--live true` is set, which means generate the data on the fly instead. 

To see if the output is reasonable we can run

```python 90_session_plotter.py --dataset sessions-reference --session-idx 0 --steps 512```

Which plots the full first (idx=0) session.

![session 0](../../images/session_reference_example0.gif)

or second (idx=1) session.

![session 1](../../images/session_reference_example1.gif)


## Generate some basic data
python 01_generate_data.py

## Load and visualize the first session (png and gif output)
python 90_session_plotter.py --session ./sessions-default/session_00000000.pkl 

## Train a model
python 12_task2_model_training.py --dataset ./sessions-default

## Generate more data
bash 00_setup.sh
