from rf import ULADetector, NoiseWrapper, QAMSource, beamformer
import os
import numpy as np
import pickle
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--carrier-frequency', type=float, required=False, default=2.4e9)
parser.add_argument('--signal-frequency', type=float, required=False, default=100e3)
parser.add_argument('--sampling-frequency', type=float, required=False, default=10e6)
parser.add_argument('--width', type=int, required=False, default=256)
parser.add_argument('--height', type=int, required=False, default=256)
parser.add_argument('--seed', type=int, required=False, default=0)
parser.add_argument('--time-steps', type=int, required=False, default=100)
parser.add_argument('--time-interval', type=float, required=False, default=0.1)
parser.add_argument('--samples', type=int, required=False, default=32)
parser.add_argument('--sessions', type=int, required=False, default=1024)
parser.add_argument('--output', type=str, required=False, default="sessions")


args = parser.parse_args()

os.mkdir(args.output)

c=3e8 # speed of light
wavelength=c/args.carrier_frequency
np.random.seed(seed=args.seed)

#setup 
def time_to_detector_offset(t,orbital_width,orbital_height,orbital_frequency=1/300.0,phase_offset=0): #1/2.0):
	x=np.cos( 2*np.pi*orbital_frequency*t+phase_offset)*orbital_width
	y=np.sin( 2*np.pi*orbital_frequency*t+phase_offset)*orbital_height
	return np.array([
		args.width/2+x,
		args.height/2+y])

for session_idx in np.arange(args.sessions):
	print("Session %d" % session_idx)
	d=ULADetector(args.sampling_frequency,2,wavelength/2) # 10Mhz sampling

	source_position=np.hstack([
		np.random.randint(low=0, high=args.width,size=(1,)),
		np.random.randint(low=0, high=args.height,size=(1,)) ])

	print("	Source position",source_position)
	phase_offset=np.random.uniform(0,2*np.pi)
	print("	Phase-offset",phase_offset)


	sigma=0
	d.add_source(NoiseWrapper(
	  QAMSource(
		source_position, # x, y position
		args.carrier_frequency,
		args.signal_frequency,
		sigma=sigma),
	  sigma=sigma))

	d.position_offset=0

	# lets run this thing
	source_positions=[ list() for _ in range(len(d.sources))]
	receiver_positions=[ list() for _ in range(len(d.receivers))]
	signal_matrixs=[]
	beam_former_outputs=[]


	for t_idx in np.arange(args.time_steps):
		t=args.time_interval*t_idx
		d.position_offset=time_to_detector_offset(t=t,orbital_width=args.width,orbital_height=args.height,phase_offset=phase_offset)
		source_positions[0].append(d.sources[0].pos)
		receiver_positions[0].append(d.receiver_pos(0))
		receiver_positions[1].append(d.receiver_pos(1))
		sm=d.get_signal_matrix(start_time=t,duration=args.samples/d.sampling_frequency)
		signal_matrixs.append(sm[None,:])
		beam_former_outputs.append(
			beamformer(d,sm,args.carrier_frequency,spacing=256+1)[1].reshape(1,-1))
	session={
			'source_positions':[ np.vstack(x) for x in source_positions],
			'receiver_positions':[ np.vstack(x) for x in receiver_positions],
			'signal_matrixs':np.vstack(signal_matrixs),
			'beam_former_outputs':np.vstack(beam_former_outputs),
			'phase_offset':phase_offset
		}
	pickle.dump(session,open("/".join([args.output,'session_%08d.pkl' % session_idx]),'wb'))
