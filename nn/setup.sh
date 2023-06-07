rm -rf sessions_task1 sessions_task2

#generate new data
n=8192
width=128
array_type="circular"
elements=4
sigma=0.0
python generate_data.py --sources 1 --output sessions_task1 --session $n --width $width --height $width --array-type $array_type --elements $elements --sigma $sigma
python generate_data.py --sources 2 --output sessions_task2 --session $n --width $width --height $width --array-type $array_type --elements $elements --sigma $sigma

#sanity check the data
python rf_torch.py

#train networks
#python task1_simple_network.py
