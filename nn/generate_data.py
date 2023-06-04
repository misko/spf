from rf import ULADetector, NoiseWrapper, QAMSource, beamformer
import os
import numpy as np
import pickle
import sys
import argparse

c=3e8 # speed of light

#setup 
def time_to_detector_offset(t,orbital_width,orbital_height,orbital_frequency=1/100.0,phase_offset=0): #1/2.0):
	x=np.cos( 2*np.pi*orbital_frequency*t+phase_offset)*orbital_width
	y=np.sin( 2*np.pi*orbital_frequency*t+phase_offset)*orbital_height
	return np.array([
		args.width/2+x,
		args.height/2+y])

def generate_session(args_and_session_idx):
	args,session_idx=args_and_session_idx
	np.random.seed(seed=args.seed+session_idx)
	wavelength=c/args.carrier_frequency
	d=ULADetector(args.sampling_frequency,2,wavelength/2) # 10Mhz sampling

	fixed_source_positions=np.random.randint(low=0, high=args.width,size=(args.sources,2))

	source_IQ=np.random.uniform(0,2*np.pi,size=(args.sources,2))

	detector_position_phase_offset=np.random.uniform(0,2*np.pi)

	sigma=0
	d.position_offset=0

	# lets run this thing
	time_stamps=[]
	source_positions_at_t=[]
	broadcasting_positions_at_t=[]
	receiver_positions_at_t=[]
	signal_matrixs_at_t=[]
	beam_former_outputs_at_t=[]
	detector_position_phase_offsets_at_t=[]

	for t_idx in np.arange(args.time_steps):
		t=args.time_interval*t_idx
		time_stamps.append(t)

		#only one source transmits at a time, TDM this part
		tdm_source_idx=np.random.randint(0,args.sources)
		d.rm_sources()
		d.add_source(NoiseWrapper(
		  QAMSource(
			fixed_source_positions[tdm_source_idx], # x, y position
			args.carrier_frequency,
			args.signal_frequency,
			sigma=sigma,
			IQ=source_IQ[tdm_source_idx]),
		  sigma=sigma))
		broadcasting_positions_at_t.append(fixed_source_positions[tdm_source_idx])

		#set the detector position (its moving)
		d.position_offset=time_to_detector_offset(t=t,orbital_width=args.width/4,orbital_height=args.height/3,phase_offset=detector_position_phase_offset)

		detector_position_phase_offsets_at_t.append(detector_position_phase_offset)
		source_positions_at_t.append(fixed_source_positions)
		receiver_positions_at_t.append(
			np.array([ d.receiver_pos(idx) for idx in np.arange(d.n_receivers()) ])) #
		sm=d.get_signal_matrix(start_time=t,duration=args.samples_per_snapshot/d.sampling_frequency)
		signal_matrixs_at_t.append(sm[None,:])
		beam_former_outputs_at_t.append(
			beamformer(d,sm,args.carrier_frequency,spacing=256+1)[1].reshape(1,-1))	
	session={
			'broadcasting_positions_at_t':broadcasting_positions_at_t, # list of (n_broadcasting,2[x,y]) 
			'source_positions_at_t':np.concatenate([ np.vstack(x)[None] for x in source_positions_at_t],axis=0), # (time_steps,sources,2[x,y])
			'receiver_positions_at_t':np.concatenate([ np.vstack(x)[None] for x in receiver_positions_at_t],axis=0), # (time_steps,receivers,2[x,y])
			'signal_matrixs_at_t':np.vstack(signal_matrixs_at_t), # (time_steps,receivers,samples_per_snapshot)
			'beam_former_outputs_at_t':np.vstack(beam_former_outputs_at_t), #(timesteps,thetas_tested_for_steering)
			'detector_position_phase_offsets_at_t':detector_position_phase_offsets_at_t,
			'time_stamps':np.vstack(time_stamps)
		}
	pickle.dump(session,open("/".join([args.output,'session_%08d.pkl' % session_idx]),'wb'))

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--carrier-frequency', type=float, required=False, default=2.4e9)
	parser.add_argument('--signal-frequency', type=float, required=False, default=100e3)
	parser.add_argument('--sampling-frequency', type=float, required=False, default=10e6)
	parser.add_argument('--sources', type=int, required=False, default=2)
	parser.add_argument('--width', type=int, required=False, default=256)
	parser.add_argument('--height', type=int, required=False, default=256)
	parser.add_argument('--seed', type=int, required=False, default=0)
	parser.add_argument('--time-steps', type=int, required=False, default=100)
	parser.add_argument('--time-interval', type=float, required=False, default=5.0)
	parser.add_argument('--samples-per-snapshot', type=int, required=False, default=3)
	parser.add_argument('--sessions', type=int, required=False, default=1024)
	parser.add_argument('--output', type=str, required=False, default="sessions-multisource")
	parser.add_argument('--cpus', type=int, required=False, default=8)
	args = parser.parse_args()
	os.mkdir(args.output)
	pickle.dump(args,open("/".join([args.output,'args.pkl']),'wb'))

	from joblib import Parallel, delayed  
	from tqdm import tqdm  
	result = Parallel(n_jobs=args.cpus)(delayed(generate_session)((args,session_idx)) for session_idx in tqdm(range(args.sessions)))
