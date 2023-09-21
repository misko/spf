import matplotlib.pyplot as plt
import numpy as np
#from numba import jit
import functools
numba=False

'''

Given some guess of the source of direction we can shift the carrier frequency
phase of received samples at the N different receivers. If the guess of the
source direction is correct, the signal from the N different receivers should
interfer constructively.

'''
@functools.lru_cache(maxsize=1024)
def rf_linspace(s,e,i):
    return np.linspace(s,e,i)

@functools.lru_cache(maxsize=1024)
def rotation_matrix(orientation):
  s = np.sin(orientation)
  c = np.cos(orientation)
  return np.array([c, -s, s, c]).reshape(2,2)

c=3e8 # speed of light


class Source(object):
  def __init__(self,pos):
    self.pos=np.array(pos)

  def signal(self,sampling_times):
    return np.cos(2*np.pi*sampling_times)+np.sin(2*np.pi*sampling_times)*1j

  def demod_signal(self,signal,demod_times):
    return signal

class SinSource(Source):
  def __init__(self,pos,frequency,phase=0,amplitude=1):
    super().__init__(pos)
    self.frequency=frequency
    self.phase=phase
    self.amplitude=amplitude

  def signal(self,sampling_times):
    #return np.cos(2*np.pi*sampling_times*self.frequency+self.phase)+np.sin(2*np.pi*sampling_times*self.frequency+self.phase)*1j
    return (self.amplitude*np.sin(2*np.pi*sampling_times*self.frequency+self.phase))#.reshape(1,-1)

class MixedSource(Source):
  def __init__(self,source_a,source_b,h=None):
    super().__init__(pos)
    self.source_a=source_a
    self.source_b=source_b
    self.h=h

  def signal(self,sampling_times):
    return self.source_a(sampling_times)*self.source_b(sampling_times)

class CarrierSource(Source):
  def __init__(self,pos,carrier_frequency,if_source,phase=0,amplitude=1,h=None):
    super().__init__(pos)
    self.carrier_frequency=carrier_frequency
    self.carrier_source=SinSource(pos,carrier_frequency,phase,amplitude)
    self.if_source=if_source
    self.h=h

  def signal(self,sampling_times):
    return (self.carrier_source.signal(sampling_times)*self.if_source.signal(sampling_times)) #.reshape(1,-1)

  def demod_signal(self,signal,demod_times,use_filter=True,nested=False):
    signal_if=signal*self.carrier_source.signal(demod_times)
    if self.h is not None and use_filter:
      signal_if=np.array([np.convolve(x, self.h, mode='same') for x in signal_if ])
    if not nested:
      return signal_if
    return self.if_source.demod_signal(signal_if,demod_times,use_filter=use_filter)
     

class QAMSource(Source):
  def __init__(self,pos,signal_frequency,sigma=0,IQ=(0,0),h=None):
    super().__init__(pos)
    self.lo_in_phase=SinSource(pos,signal_frequency,-np.pi/2) #,amplitude=IQ[0]) # cos
    self.lo_out_of_phase=SinSource(pos,signal_frequency,0) #,amplitude=IQ[1]) # sin
    self.sigma=sigma
    self.IQ=IQ
    self.h=h

  def signal(self,sampling_times):
    signal=((self.lo_in_phase.signal(sampling_times)*self.IQ[0]+self.lo_out_of_phase.signal(sampling_times)*self.IQ[1])/2)
    #signal+=(np.random.randn(sampling_times.shape[0])*self.sigma) #.view(np.cdouble).reshape(-1)
    return signal #.reshape(1,-1)

  def demod_signal(self,signal,demod_times,use_filter=True):
    _i=signal*self.lo_in_phase.signal(demod_times)
    _q=signal*self.lo_out_of_phase.signal(demod_times)
    if self.h is not None and use_filter:
      _i=np.array([np.convolve(x,self.h,mode='same') for x in _i ])
      _q=np.array([np.convolve(x,self.h,mode='same') for x in _q ])
    return _i+1j*_q

class NoiseWrapper(Source):
  def __init__(self,internal_source,sigma=1):
    super().__init__(internal_source.pos)
    self.internal_source=internal_source
    self.sigma=sigma

  def signal(self,sampling_times):
    return self.internal_source.signal(sampling_times) + (np.random.randn(sampling_times.shape[0], 2)*self.sigma).view(np.cdouble).reshape(-1)

class Detector(object):
  def __init__(self,sampling_frequency,oreintation=0,sigma=0.0):
    self.sources=[]
    self.source_positions=None
    self.receiver_positions=None
    self.sampling_frequency=sampling_frequency
    self.position_offset=np.zeros(2)
    self.orientation=0.0
    self.sigma=sigma

  def add_source(self,source):
    self.sources.append(source)
    if self.source_positions is None:
      self.source_positions=np.array(source.pos).reshape(1,2)
    else:
      self.source_positions=np.vstack([
            self.source_positions,
            np.array(source.pos).reshape(1,2)])

  def distance_receiver_to_source(self):
    return np.linalg.norm(self.all_receiver_pos()[:,None]-self.source_positions[None],axis=2)

  def rm_sources(self):
    self.sources=[]
    self.source_positions=None

  def set_receiver_positions(self,receiver_positions):
    self.receiver_positions=receiver_positions

  def add_receiver(self,receiver_position):
    if self.receiver_positions is None:
      self.receiver_positions=np.array(receiver_position).reshape(1,2)
    else:
      self.receiver_positions=np.vstack([
            self.receiver_positions,
            np.array(receiver_position).reshape(1,2)])

  def n_receivers(self):
    return self.receiver_positions.shape[0]


  def all_receiver_pos(self):
    return self.position_offset+(self.receiver_positions @ rotation_matrix(self.orientation))

  def receiver_pos(self,receiver_idx):
    return self.position_offset+(self.receiver_positions[receiver_idx] @ rotation_matrix(self.orientation))

  def get_signal_matrix_old(self,start_time,duration,rx_lo=0):
    n_samples=int(duration*self.sampling_frequency)
    base_times=start_time+rf_linspace(0,n_samples-1,n_samples)/self.sampling_frequency
    sample_matrix=np.zeros((self.n_receivers(),n_samples),dtype=np.cdouble) # receivers x samples
    for receiver_index,receiver in enumerate(self.receiver_positions):
      for _source in self.sources:
        distance=np.linalg.norm(self.receiver_pos(receiver_index)-_source.pos)
        time_delay=distance/c
        sample_matrix[receiver_index,:]+=_source.demod_signal(_source.signal(base_times-time_delay)/(distance**2),
                                                              base_times)
        if rx_lo>0:
          sample_matrix[receiver_index,:]
    return sample_matrix

  def get_signal_matrix(self,start_time,duration,rx_lo=0):
    n_samples=int(duration*self.sampling_frequency)
    base_times=start_time+rf_linspace(0,n_samples-1,n_samples)/self.sampling_frequency
    #sample_matrix=np.zeros((self.receiver_positions.shape[0],n_samples),dtype=np.cdouble) # receivers x samples
    sample_matrix=np.random.randn(
        self.receiver_positions.shape[0],n_samples,2).view(np.cdouble).reshape(self.receiver_positions.shape[0],n_samples)*self.sigma

    if len(self.sources)==0:
      return sample_matrix

    distances=self.distance_receiver_to_source().T # sources x receivers # TODO numerical stability,  maybe project angle and calculate diff
    #TODO diff can be small relative to absolute distance
    time_delays=distances/c 
    base_time_offsets=base_times[None,None]-(distances/c)[...,None] # sources x receivers x sampling intervals
    distances_squared=distances**2
    raw_signal=[]
    for source_index,_source in enumerate(self.sources):
      #get the signal from the source for these times
      print(_source)
      signal=_source.signal(base_time_offsets[source_index]) #.reshape(base_time_offsets[source_index].shape) # receivers x sampling intervals
      raw_signal.append(signal)
      normalized_signal=signal/distances_squared[source_index][...,None]
      _base_times=np.broadcast_to(base_times,normalized_signal.shape) # broadcast the basetimes for rx_lo on all receivers
      ds=_source.demod_signal(
              normalized_signal,
              _base_times,nested=True)
      sample_matrix+=ds
    return sample_matrix,raw_signal


@functools.lru_cache(maxsize=1024)
def linear_receiver_positions(n_elements,spacing):
    receiver_positions=np.zeros((n_elements,2))
    receiver_positions[:,0]=spacing*(np.arange(n_elements)-(n_elements-1)/2)
    return receiver_positions

class ULADetector(Detector):
  def __init__(self,sampling_frequency,n_elements,spacing,sigma=0.0):
    super().__init__(sampling_frequency,sigma=sigma)
    self.set_receiver_positions(linear_receiver_positions(n_elements,spacing))
    
@functools.lru_cache(maxsize=1024)
def circular_receiver_positions(n_elements,radius):
    theta=(rf_linspace(0,2*np.pi,n_elements+1)[:-1]+np.pi/2).reshape(-1,1)
    return radius*np.hstack([np.cos(theta),np.sin(theta)])

class UCADetector(Detector):
  def __init__(self,sampling_frequency,n_elements,radius,sigma=0.0):
    super().__init__(sampling_frequency,sigma=sigma)
    self.set_receiver_positions(circular_receiver_positions(n_elements,radius))

@functools.lru_cache(maxsize=1024)
def get_thetas(spacing):
    thetas=rf_linspace(-np.pi,np.pi,spacing)
    return thetas,np.vstack([np.cos(thetas)[None],np.sin(thetas)[None]]).T

if numba:
    @jit(nopython=True)
    def beamformer_numba_helper(receiver_positions,signal_matrix,carrier_frequency,spacing,thetas,source_vectors):
        steer_dot_signal=np.zeros(thetas.shape[0])
        carrier_wavelength=c/carrier_frequency

        projection_of_receiver_onto_source_directions=(source_vectors @ receiver_positions.T)
        args=2*np.pi*projection_of_receiver_onto_source_directions/carrier_wavelength
        steering_vectors=np.exp(-1j*args)
        steer_dot_signal=np.absolute(steering_vectors @ signal_matrix).sum(axis=1)/signal_matrix.shape[1]

        return thetas,steer_dot_signal,steering_vectors


def beamformer_numba(receiver_positions,signal_matrix,carrier_frequency,spacing=64+1):
    thetas,source_vectors=get_thetas(spacing)
    return beamformer_numba_helper(receiver_positions,
            signal_matrix,
            carrier_frequency,
            spacing,
            thetas,source_vectors)

#from Jon Kraft github
def dbfs(raw_data):
    # function to convert IQ samples to FFT plot, scaled in dBFS
    NumSamples = len(raw_data)
    win = np.hamming(NumSamples)
    y = raw_data * win
    s_fft = np.fft.fft(y) / np.sum(win)
    s_shift = np.fft.fftshift(s_fft)
    s_dbfs = 20*np.log10(np.abs(s_shift)/(2**11))     # Pluto is a signed 12 bit ADC, so use 2^11 to convert to dBFS
    return s_dbfs

def beamformer(receiver_positions,signal_matrix,carrier_frequency,calibration=None,spacing=64+1,offset=0.0):
    thetas=np.linspace(-np.pi,np.pi,spacing)#-offset
    source_vectors=np.vstack([np.sin(thetas+offset)[None],np.cos(thetas+offset)[None]]).T
    #thetas,source_vectors=get_thetas(spacing)
    steer_dot_signal=np.zeros(thetas.shape[0])
    carrier_wavelength=c/carrier_frequency

    projection_of_receiver_onto_source_directions=(source_vectors @ receiver_positions.T)
    args=2*np.pi*projection_of_receiver_onto_source_directions/carrier_wavelength
    steering_vectors=np.exp(-1j*args)

    if calibration is not None:
      steering_vectors=steering_vectors*calibration[None]
    #the delay sum is performed in the matmul step, the absolute is over the summed value
    steer_dot_signal=np.absolute(np.matmul(steering_vectors,signal_matrix)).mean(axis=1)
    return thetas,steer_dot_signal,steering_vectors

def beamformer_old(receiver_positions,signal_matrix,carrier_frequency,calibration=None,spacing=64+1):
    if calibration is None:
        calibration=np.ones(receiver_positions.shape[0]).astype(np.cdouble)
    thetas=rf_linspace(-np.pi,np.pi,spacing)
    source_vectors=np.vstack([np.cos(thetas)[None],np.sin(thetas)[None]]).T
    steer_dot_signal=np.zeros(thetas.shape[0])
    carrier_wavelength=c/carrier_frequency
    steering_vectors=np.zeros((len(thetas),receiver_positions.shape[0])).astype(np.cdouble)
    for theta_index,theta in enumerate(thetas):
        source_vector=np.array([np.cos(theta),np.sin(theta)])
        for receiver_index in np.arange(receiver_positions.shape[0]):
            projection_of_receiver_onto_source_direction=np.dot(source_vector,receiver_positions[receiver_index])
            arg=2*np.pi*projection_of_receiver_onto_source_direction/carrier_wavelength
            steering_vectors[theta_index][receiver_index]=np.exp(-1j*arg)
        #the delay sum is performed in the matmul step, the absolute is over the summed value
        steer_dot_signal[theta_index]=np.absolute(np.matmul(steering_vectors[theta_index]*calibration,signal_matrix)).mean()

    return thetas,steer_dot_signal,steering_vectors
  
