#rm -rf sessions_task1 sessions_task2

#generate new data
width=128
array_type="circular"
elements=4
time_steps=256
time_interval=0.3
for n in $((2**19)); do
for sigma in 0.0 0.1 0.5; do
	for sources in 3 2 1; do
	python 01_generate_data.py \
		--output sessions_sigma${sigma}_sources${sources}_n${n}_timesteps${time_steps}_timeinterval${time_interval}_dtbounce --detector-trajectory bounce \
		--session $n --width $width --array-type $array_type --elements $elements --sigma $sigma \
		--time-steps ${time_steps} --time-interval ${time_interval} --detector-speed 5 --sources ${sources}
done
done	
done



#sanity check the data
#python rf_torch.py

#train networks
#python task1_simple_network.py
