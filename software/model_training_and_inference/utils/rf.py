import matplotlib.pyplot as plt
import numpy as np

'''

Given some guess of the source of direction we can shift the carrier frequency
phase of received samples at the N different receivers. If the guess of the
source direction is correct, the signal from the N different receivers should
interfer constructively.

'''

c=3e8 # speed of light

class Source(object):
  def __init__(self,pos):
    self.pos=np.array(pos)

  def signal(self,sampling_times):
    return np.cos(2*np.pi*sampling_times)+np.sin(2*np.pi*sampling_times)*1j

  def demod_signal(self,signal,demod_times):
    return signal

class SinSource(Source):
  def __init__(self,pos,frequency,phase):
    super().__init__(pos)
    self.frequency=frequency
    self.phase=phase

  def signal(self,sampling_times):
    return np.cos(2*np.pi*sampling_times*self.frequency+self.phase)+np.sin(2*np.pi*sampling_times*self.frequency+self.phase)*1j

class MixedSource(Source):
  def __init__(self,source_a,source_b):
    super().__init__(pos)
    self.source_a=source_a
    self.source_b=source_b

  def signal(self,sampling_times):
    return self.source_a(sampling_times)*self.source_b(sampling_times)

class QAMSource(Source):
  def __init__(self,pos,carrier_frequency,signal_frequency,sigma=0,IQ=(0,0)):
    super().__init__(pos)
    self.lo_in_phase=SinSource(pos,carrier_frequency,-np.pi/2+IQ[0]) # cos
    self.lo_out_of_phase=SinSource(pos,carrier_frequency,0+IQ[1]) # cos
    self.signal_source=SinSource(pos,signal_frequency,0)
    self.sigma=sigma

  def signal(self,sampling_times):
    signal=self.signal_source.signal(sampling_times)
    return ((self.lo_in_phase.signal(sampling_times)*signal.real+\
              self.lo_out_of_phase.signal(sampling_times)*signal.imag)/2) +\
                np.random.randn(sampling_times.shape[0])*self.sigma

  def demod_signal(self,signal,demod_times):
    return (self.lo_in_phase(demod_times)+\
              self.lo_out_of_phase(demod_times)*1j)*signal
  

class NoiseWrapper(Source):
  def __init__(self,internal_source,sigma=1):
    super().__init__(internal_source.pos)
    self.internal_source=internal_source
    self.sigma=sigma

  def signal(self,sampling_times):
    return self.internal_source.signal(sampling_times) + (np.random.randn(sampling_times.shape[0], 2)*self.sigma).view(np.cdouble).reshape(-1)

class Receiver:
  def __init__(self,pos):
    self.pos=np.array(pos)

class Detector(object):
  def __init__(self,sampling_frequency,oreintation=0):
    self.sources=[]
    self.receivers=[]
    self.sampling_frequency=sampling_frequency
    self.position_offset=np.zeros(2)
    self.orientation=0.0

  def add_source(self,source):
    self.sources.append(source)

  def rm_sources(self):
    self.sources=[]

  def add_receiver(self,receiver):
    self.receivers.append(receiver)

  def n_receivers(self):
    return len(self.receivers)

  def rotation_matrix(self):
    s = np.sin(self.orientation)
    c = np.cos(self.orientation)
    return np.stack([np.stack([c, -s]),
		   np.stack([s, c])])

  def receiver_pos(self,receiver_idx):
    return self.position_offset+(self.receivers[receiver_idx].pos @ self.rotation_matrix())

  def get_signal_matrix(self,start_time,duration,rx_lo=0):
    n_samples=int(duration*self.sampling_frequency)
    base_times=start_time+np.linspace(0,n_samples-1,n_samples)/self.sampling_frequency
    sample_matrix=np.zeros((len(self.receivers),n_samples),dtype=np.cdouble) # receivers x samples
    for receiver_index,receiver in enumerate(self.receivers):
      for _source in self.sources:
        distance=np.linalg.norm(self.receiver_pos(receiver_index)-_source.pos)
        time_delay=distance/c
        sample_matrix[receiver_index,:]+=_source.demod_signal(_source.signal(base_times-time_delay)/(distance**2),
                                                              base_times)
        if rx_lo>0:
          sample_matrix[receiver_index,:]
    return sample_matrix

 

class ULADetector(Detector):
  def __init__(self,sampling_frequency,n_elements,spacing):
    super().__init__(sampling_frequency)
    for idx in np.arange(n_elements):
      self.add_receiver(Receiver([
          spacing*(idx-(n_elements-1)/2),
          0]))
      
class UCADetector(Detector):
  def __init__(self,sampling_frequency,n_elements,radius):
    super().__init__(sampling_frequency)
    for theta in np.linspace(0,2*np.pi,n_elements+1)[:-1]+np.pi/2: # orientate along y axis
      self.add_receiver(Receiver([
          radius*np.cos(theta),
          radius*np.sin(theta)]))

def beamformer(detector,signal_matrix,carrier_frequency,calibration=None,spacing=64+1):
    if calibration is None:
        calibration=np.ones(len(detector.receivers)).astype(np.cdouble)
    thetas=np.linspace(-np.pi,np.pi,spacing)
    steer_dot_signal=np.zeros(thetas.shape[0])
    carrier_wavelength=c/carrier_frequency
    steering_vectors=np.zeros((len(thetas),len(detector.receivers))).astype(np.cdouble)
    for theta_index,theta in enumerate(thetas):
        source_vector=np.array([np.cos(theta),np.sin(theta)])
        projections=[]
        for receiver_index,receiver in enumerate(detector.receivers):
            projection_of_receiver_onto_source_direction=np.dot(source_vector,detector.receiver_pos(receiver_index))
            projections.append(projection_of_receiver_onto_source_direction/carrier_wavelength)
            arg=2*np.pi*projection_of_receiver_onto_source_direction/carrier_wavelength
            steering_vectors[theta_index][receiver_index]=np.exp(-1j*arg)
        steer_dot_signal[theta_index]=np.absolute(np.matmul(steering_vectors[theta_index]*calibration,signal_matrix)).mean()
    return thetas,steer_dot_signal,steering_vectors
  
