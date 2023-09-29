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

## Raw dataset contents

- broadcasting_positions_at_t, (time_steps,1), an int representing which source is broadcasting at any single time
- source_positions_at_t, (time_steps,nsources,2), 2 floats (xy) for each source, representing position of the source at t
- source_velocities_at_t, (time_steps,nsources,2), 2 floats (xy) for each source, representing velocity of the source at t
- receiver_positions_at_t, (time_steps,nreceivers,2), 2 floats (xy) for each receiver, representing the position of the receiver at t
- signal_matrixs_at_t, (time_steps,nreceivers,samples_per_snapshot), floats for each receiver representing the signal recieved at t
- beam_former_outputs_at_t, (time_steps,nthetas), a float for every theta (direction) tested by the beamformer reprensenting power in that direction
- thetas_at_t':thetas_at_t, (time_steps,nthetas), every theta float tested for the beamformer
- detector_position_phase_offsets_at_t, (time_steps,1), detector phase offset for orbit dynamics
- time_stamps, (time_steps,1), the actual time in seconds for each time step
- width_at_t, (time_steps,1), the width (and height) of the session in meters at time t
- detector_orientation_at_t, (time_steps,1), orientation of detector, same as direction of travel, in radians
- detector_position_at_t, (time_steps,2), 2 floats (xy) describing the position of the detector at any time t
- source_theta_at_t, (time_steps,1), 1 float representing the difference in angle between the detector orientation and the broadcasting source
- source_distance_at_t':source_distance_at_t, (time_steps,1), 1 float reprensenting the distance to the source currently broadcasting

## Collated dataset for trajectory net

- drone_state, (batch,time_steps, 8) 
	- detector_position_at_t_normalized_centered, 2 cols (idx=0,1)
  - normalized_01_times, times normalized to 0...1 interval, 1 col (idx=2)
  - normalized_pirads_space_theta, units are pi*rad , orientation of relative to last xy position, movement direction, 1 col (idx=5)
  - space_delta, (width normalized) difference vector of detector from last time step to now, 2 col (idx=3,4)
  - space_dist, (width normalized) distance travelled since last time point, 1 col (idx=6)
  - normalized_pirads_detector_theta, units are pi*rad , orientation of detector relative to x/y, 1 col (idx=7)
- emitter_position_and_velocity, (batch,time_steps,nsources,4)
  - source_positions_at_t_normalized_centered
  - source_velocities_at_t_normalized
- emitters_broadcasting, (batch, time_steps, nsources,1) , which emitter is broadcasting [0/1]
- emitters_n_broadcasts, (batch, time_steps, nsourcs, 1) , how many times now+previous has this emitter broadcast
- radio_feature, (batch, timesteps, 1+len(thetas))
  - torch.log(d['beam_former_outputs_at_t'].mean(axis=2,keepdim=True))/20 , average power
  - d['beam_former_outputs_at_t']/d['beam_former_outputs_at_t'].mean(axis=2,keepdim=True), mean normalized bf outputs

## Running a simple training loop

python 14_task3_model_training.py --dataset ./sessions-reference  --mb 64 --workers 0 --plot True --snapshots 128 --save-every 10000 --type 32 --lr-transformer $lr  --print-every 256 --plot-every 2048  --output-prefix output_mb64_reference_ --device mps --n-layers 1 4

## Visualizing sessions

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
