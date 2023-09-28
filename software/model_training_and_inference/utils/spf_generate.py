
import argparse
import os
import pickle
import sys
import bz2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from compress_pickle import dump, load
from utils.rf import NoiseWrapper, IQSource, UCADetector, ULADetector, beamformer_numba, beamformer, beamformer_old

c=3e8 # speed of light

class BoundedPoint:
  def __init__(self,
    pos=np.ones(2)*0.5,
    v=np.zeros(2),
    delta_time=0.05,
    width=128):
    self.pos=pos
    self.v=v
    self.delta_time=delta_time
    self.width=width

  def time_step(self):
    if np.linalg.norm(self.v)==0:
      return np.array(self.pos),0.0,np.zeros(2)
    self.pos+=self.v*self.delta_time

    for idx in [0,1]:
      while self.pos[idx]>self.width or self.pos[idx]<0:
        self.pos=np.clip(self.pos,0,self.width)
        self.v[idx]=-self.v[idx]
    if np.linalg.norm(self.v)>0:
      return np.array(self.pos),np.arctan2(self.v[0],self.v[1]),self.v
    return np.array(self.pos),0.0,np.zeros(2)


def time_to_detector_offset(t,orbital_width,orbital_height,orbital_frequency=1/100.0,phase_offset=0): #1/2.0):
  x=np.cos( 2*np.pi*orbital_frequency*t+phase_offset)*orbital_width
  y=np.sin( 2*np.pi*orbital_frequency*t+phase_offset)*orbital_height
  return np.array([
    1/2+x,
    1/2+y])




def generate_session(args_and_session_idx):
  args,session_idx=args_and_session_idx
  np.random.seed(seed=args.seed+session_idx)
  wavelength=c/args.carrier_frequency

  n_sources=args.sources
  n_sources_used=args.sources
  if args.sources<0:
    n_sources_used=np.random.choice(np.arange(1,(-args.sources)+1))
    n_sources=-args.sources
  detector_speed=args.detector_speed
  if args.detector_speed<0:
    detector_speed=np.random.uniform(low=0.0, high=-args.detector_speed)
  sigma=args.sigma
  if args.sigma<0:
    sigma=np.random.uniform(low=0.0, high=-args.sigma)
  detector_noise=args.detector_noise
  if args.detector_noise<0:
    detector_noise=np.random.uniform(low=0.0, high=-args.detector_noise)

  beamformer_f=beamformer_numba if args.numba else beamformer

  if args.array_type=='linear':
    d=ULADetector(args.sampling_frequency,args.elements,wavelength/4,sigma=detector_noise) # 10Mhz sampling
  elif args.array_type=='circular':
    d=UCADetector(args.sampling_frequency,args.elements,wavelength/4,sigma=detector_noise) # 10Mhz sampling
  else:
    print("Array type must be linear or circular")
    sys.exit(1)

  current_source_positions=np.random.uniform(low=0, high=args.width,size=(n_sources,2))
  current_source_velocities=np.zeros((n_sources,2))

  if args.reference:
    current_source_positions=current_source_positions*0+np.array((args.width//2,args.width//4))
    sigma=0
    n_sources=1
    n_sources_used=1


  detector_position_phase_offset=np.random.uniform(0,2*np.pi)
  detector_position_phase_offsets_at_t=np.ones((args.time_steps,1))*detector_position_phase_offset

  d.position_offset=0

  # lets run this thing
  source_positions_at_t=np.zeros((args.time_steps,n_sources,2))
  source_velocities_at_t=np.zeros((args.time_steps,n_sources,2))
  broadcasting_positions_at_t=np.zeros((args.time_steps,n_sources,1))
  broadcasting_heading_at_t=np.zeros((args.time_steps,n_sources,1))
  receiver_positions_at_t=np.zeros((args.time_steps,args.elements,2))
  source_theta_at_t=np.zeros((args.time_steps,1,1))
  source_distance_at_t=np.zeros((args.time_steps,1,1))

  detector_orientation_at_t=np.ones((args.time_steps,1))

  signal_matrixs_at_t=np.zeros((args.time_steps,args.elements,args.samples_per_snapshot),dtype=np.complex64)
  beam_former_outputs_at_t=np.zeros((args.time_steps,args.beam_former_spacing))
  detector_position_at_t=np.zeros((args.time_steps,2))

  thetas_at_t=np.zeros((args.time_steps,args.beam_former_spacing))

  detector_theta=np.random.uniform(-np.pi,np.pi)
  if args.reference:
    detector_theta=np.random.choice([0,np.pi/4,np.pi*3/4,np.pi/2,np.pi])
    #detector_theta=np.random.choice([0,np.pi/4,np.pi/2,np.pi])

  detector_v=np.array([np.cos(detector_theta),np.sin(detector_theta)])*detector_speed # 10m/s
  detector_bounded_point=BoundedPoint(pos=np.random.uniform(0+10,args.width-10,2),
      v=detector_v,
      delta_time=args.time_interval)

  source_bounded_points=[]
  for idx in range(n_sources):
    source_theta=np.random.uniform(-np.pi,np.pi)
    source_speed=args.source_speed
    if source_speed<0:
      source_speed=np.random.uniform(low=0.0, high=-source_speed)
    source_v=np.array(
        [
          np.cos(source_theta),
          np.sin(source_theta)])*source_speed 
    source_bounded_points.append(
        BoundedPoint(
          pos=np.random.uniform(0+10,args.width-10,2),
          v=source_v,
          delta_time=args.time_interval)
      )

  whos_broadcasting_at_t=np.random.randint(0,n_sources_used,args.time_steps)

  if args.random_emitter_timing:
    emitter_p=np.random.randint(1,10,n_sources_used)
    emitter_p=emitter_p/emitter_p.sum()
    whos_broadcasting_at_t=np.random.choice(np.arange(n_sources_used),args.time_steps,p=emitter_p)

  if args.random_silence:
    silence_p=np.random.uniform(0.0,0.8)
    whos_broadcasting_at_t[np.random.choice(np.arange(args.time_steps),int(silence_p*args.time_steps),replace=False)]=-1

  time_steps_that_broadcast=np.where(whos_broadcasting_at_t>=0)[0]
  broadcasting_positions_at_t[time_steps_that_broadcast,whos_broadcasting_at_t[time_steps_that_broadcast]]=1
  #broadcasting_positions_at_t[np.arange(args.time_steps),whos_broadcasting_at_t]=1

  #deal with source positions
  #broadcasting_source_positions=source_positions_at_t[np.where(broadcasting_positions_at_t==1)[:-1]]
   
  #deal with detector position features
  #diffs=source_positions-d['detector_position_at_t_normalized']
  #source_theta=(torch.atan2(diffs[...,1],diffs[...,0]))[:,:,None]
  #breakpoint()

  time_stamps=(np.arange(args.time_steps)*args.time_interval).reshape(-1,1)
  for t_idx in np.arange(args.time_steps):
    #update source positions
    for idx in range(len(source_bounded_points)):
      current_source_positions[idx],_,current_source_velocities[idx]=source_bounded_points[idx].time_step()
    #only one source transmits at a time, TDM this part
    tdm_source_idx=whos_broadcasting_at_t[t_idx]
    d.rm_sources()
    if tdm_source_idx>=0:
      d.add_source(NoiseWrapper(
        IQSource(
        current_source_positions[tdm_source_idx], # x, y position
        args.carrier_frequency),
        sigma=sigma))
    
    #set the detector position (its moving)
    if args.detector_trajectory=='orbit':
      print("There is an error in angles somewhere here")
      sys.exit(1)
      d.position_offset=(time_to_detector_offset(t=time_stamps[t_idx,0],
        orbital_width=1/4,
        orbital_height=1/3,
        phase_offset=detector_position_phase_offset,
        orbital_frequency=(2/3)*args.width*np.pi/detector_speed)*args.width).astype(int)
    elif args.detector_trajectory=='bounce':
      d.position_offset,d.orientation,_=detector_bounded_point.time_step()
      detector_orientation_at_t[t_idx]=d.orientation

    detector_position_phase_offsets_at_t[t_idx]=detector_position_phase_offset
    source_positions_at_t[t_idx]=current_source_positions
    source_velocities_at_t[t_idx]=current_source_velocities
    receiver_positions_at_t[t_idx]=d.all_receiver_pos()

    signal_matrixs_at_t[t_idx]=d.get_signal_matrix(
        start_time=time_stamps[t_idx,0],
        duration=args.samples_per_snapshot/d.sampling_frequency)
    thetas_at_t[t_idx],beam_former_outputs_at_t[t_idx],_=beamformer_f(
      d.all_receiver_pos(),
      signal_matrixs_at_t[t_idx],
      args.carrier_frequency,spacing=args.beam_former_spacing,
      offset=d.orientation)
    #print(d.orientation,detector_theta)
    detector_position_at_t[t_idx]=d.position_offset

    if tdm_source_idx>=0:
      diff=current_source_positions[tdm_source_idx]-detector_position_at_t[t_idx]
      source_theta_at_t[t_idx]=(np.arctan2(diff[[0]],diff[[1]])-d.orientation+np.pi)%(2*np.pi)-np.pi  
      source_distance_at_t[t_idx]=np.sqrt(np.power(diff,2).sum())
    else:
      source_theta_at_t[t_idx]=0 #(np.arctan2(diff[[1]],diff[[0]])-d.orientation+np.pi)%(2*np.pi)-np.pi  
      source_distance_at_t[t_idx]=0 #np.sqrt(np.power(diff,2).sum())
  session={
      'broadcasting_positions_at_t':broadcasting_positions_at_t, # list of (time_steps,sources,1) 
      'source_positions_at_t':source_positions_at_t, # (time_steps,sources,2[x,y])
      'source_velocities_at_t':source_velocities_at_t, # (time_steps,sources,2[x,y])
      'receiver_positions_at_t':receiver_positions_at_t, # (time_steps,receivers,2[x,y])
      'signal_matrixs_at_t':signal_matrixs_at_t, # (time_steps,receivers,samples_per_snapshot)
      'beam_former_outputs_at_t':beam_former_outputs_at_t, #(timesteps,thetas_tested_for_steering)
      'thetas_at_t':thetas_at_t, #(timesteps,thetas_tested_for_steering)
      'detector_position_phase_offsets_at_t':detector_position_phase_offsets_at_t,
      'time_stamps':time_stamps,
      'width_at_t':np.ones((args.time_steps,1),dtype=int)*args.width,
      'detector_orientation_at_t':detector_orientation_at_t,
      'detector_position_at_t':detector_position_at_t, # (time_steps,2[x,y])
      'source_theta_at_t':source_theta_at_t,
      'source_distance_at_t':source_distance_at_t,
  }
  return session

def generate_session_and_dump(args_and_session_idx):
  args,session_idx=args_and_session_idx
  session=generate_session(args_and_session_idx)
  dump(session,"/".join([args.output,'session_%08d.pkl' % session_idx]),compression="lzma")

