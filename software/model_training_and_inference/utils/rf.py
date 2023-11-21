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

'''
Rotate by orientation
If we are left multiplying then its a left (counter clockwise) rotation

'''
@functools.lru_cache(maxsize=1024)
def rotation_matrix(orientation): 
  s = np.sin(orientation)
  c = np.cos(orientation)
  return np.array([c, s, -s, c]).reshape(2,2)

c=3e8 # speed of light


class Source(object):
  def __init__(self,pos):
    self.pos=np.array(pos)
    assert(self.pos.shape[1]==2)

  def signal(self,sampling_times):
    return np.cos(2*np.pi*sampling_times)+np.sin(2*np.pi*sampling_times)*1j

  def demod_signal(self,signal,demod_times):
    return signal

class IQSource(Source):
  def __init__(self,pos,frequency,phase=0,amplitude=1):
    super().__init__(pos)
    self.frequency=frequency
    self.phase=phase
    self.amplitude=amplitude

  def signal(self,sampling_times):
    return np.cos(2*np.pi*sampling_times*self.frequency+self.phase)+np.sin(2*np.pi*sampling_times*self.frequency+self.phase)*1j

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

class NoiseWrapper(Source):
  def __init__(self,internal_source,sigma=1):
    super().__init__(internal_source.pos)
    self.internal_source=internal_source
    self.sigma=sigma

  def signal(self,sampling_times):
    assert(sampling_times.ndim==2) # receivers x time
    return self.internal_source.signal(sampling_times) + (np.random.randn(*sampling_times.shape)+np.random.randn(*sampling_times.shape)*1j)*self.sigma

class Detector(object):
  def __init__(self,sampling_frequency,orientation=0,sigma=0.0):
    self.sources=[]
    self.source_positions=None
    self.receiver_positions=None
    self.sampling_frequency=sampling_frequency
    self.position_offset=np.zeros(2)
    self.orientation=orientation # rotation to the right in radians to apply to receiver array coordinate system
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

  def all_receiver_pos(self,with_offset=True):
    if with_offset:
      return self.position_offset+(rotation_matrix(self.orientation) @ self.receiver_positions.T).T
    else:
      return (rotation_matrix(self.orientation) @ self.receiver_positions.T).T

  def receiver_pos(self,receiver_idx, with_offset=True):
    if with_offset:
      return self.position_offset+(rotation_matrix(self.orientation) @ self.receiver_positions[receiver_idx].T).T
    else:
      return (rotation_matrix(self.orientation) @ self.receiver_positions[receiver_idx].T).T

  def get_signal_matrix(self,start_time,duration,rx_lo=0):
    n_samples=int(duration*self.sampling_frequency)
    base_times=start_time+rf_linspace(0,n_samples-1,n_samples)/self.sampling_frequency

    if self.sigma==0.0:
      sample_matrix=np.zeros((self.receiver_positions.shape[0],n_samples),dtype=np.cdouble) # receivers x samples
    else:
      sample_matrix=np.random.randn(
        self.receiver_positions.shape[0],n_samples,2).view(np.cdouble).reshape(self.receiver_positions.shape[0],n_samples)*self.sigma

    if len(self.sources)==0:
      return sample_matrix

    distances=self.distance_receiver_to_source().T # sources x receivers # TODO numerical stability,  maybe project angle and calculate diff
    #TODO diff can be small relative to absolute distance
    time_delays=distances/c 
    base_time_offsets=base_times[None,None]-(distances/c)[...,None] # sources x receivers x sampling intervals
    distances_squared=distances**2
    for source_index,_source in enumerate(self.sources):
      #get the signal from the source for these times
      signal=_source.signal(base_time_offsets[source_index]) #.reshape(base_time_offsets[source_index].shape) # receivers x sampling intervals
      normalized_signal=signal/distances_squared[source_index][...,None]
      _base_times=np.broadcast_to(base_times,normalized_signal.shape) # broadcast the basetimes for rx_lo on all receivers
      demod_times=np.broadcast_to(_base_times.mean(axis=0,keepdims=True),_base_times.shape) #TODO this just takes the average?
      ds=_source.demod_signal(
              normalized_signal,
              demod_times) # TODO nested demod?
      #print(_base_times.shape,"BT")
      sample_matrix+=ds
    return sample_matrix #,raw_signal,demod_times,base_time_offsets[0]


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
    theta=(rf_linspace(0,2*np.pi,n_elements+1)[:-1]).reshape(-1,1)
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


###
'''
Beamformer takes as input the 
  receiver positions
  signal marix representing signal received at those positions
  carrier_frequency
  calibration between receivers
  spacing of thetas
  offset (subtracted from thetas, often needs to be supplied with (-) to counter the actual angle)

(1) compute spacing different directions in the unit circle to test signal strength at
(2) compute the source unit vectors for each of the directions in (1)
(3) project the receiver positions onto source unit vectors, this tells us relative distance to each receiver
(4) using the above distances, normalize to wavelength units, and compute phase adjustments
(5) transform phase adjustments into complex matrix 
(6) apply adjustment matrix to signals and take the mean of the absolute values

**Beamformer theta output is right rotated (clockwise)**

Beamformer assumes,
0 -> x=0, y=1
pi/2 -> x=1, y=0
-pi/2 -> x=-1, y=0
'''
###
def beamformer(receiver_positions,signal_matrix,carrier_frequency,calibration=None,spacing=64+1,offset=0.0):
    thetas=np.linspace(-np.pi,np.pi,spacing)#-offset
    source_vectors=np.vstack([np.sin(thetas+offset)[None],np.cos(thetas+offset)[None]]).T

    projection_of_receiver_onto_source_directions=(source_vectors @ receiver_positions.T)

    carrier_wavelength=c/carrier_frequency
    args=2*np.pi*projection_of_receiver_onto_source_directions/carrier_wavelength
    steering_vectors=np.exp(-1j*args)
    if calibration is not None:
      steering_vectors=steering_vectors*calibration[None]
    #the delay sum is performed in the matmul step, the absolute is over the summed value
    phase_adjusted=np.matmul(steering_vectors,signal_matrix) # this is adjust and sum in one step
    steer_dot_signal=np.absolute(phase_adjusted).mean(axis=1) # mean over samples
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
  
