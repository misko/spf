from utils.rf import *
c=3e8 # speed of light
carrier_frequency=2.4e9
wavelength=c/carrier_frequency
sampling_frequency=10e6
d=ULADetector(sampling_frequency,2,wavelength/4,sigma=0.0) 


source_pos=np.array([[10,0]])

signal_matrixs=[]
beamformer_outputs=[]
rotations=[-np.pi,-np.pi/2,-np.pi/4,0,np.pi/2,np.pi/2,np.pi]
for rotation in rotations:
    rot_mat=rotation_matrix(rotation)
    _source_pos=source_pos @ rot_mat

    d.rm_sources()
    d.add_source(
      QAMSource(
          _source_pos, # x, y position
          carrier_frequency,100e3))
    d.orientation=rotation
    print(rotation)
    #print("\tSRC",_source_pos)
    #print("\tREC",d.all_receiver_pos())
    signal_matrix=d.get_signal_matrix(
        start_time=100,
        duration=3/d.sampling_frequency)
    #print(signal_matrix)
    thetas_at_t,beam_former_outputs_at_t,_=beamformer(
            d.all_receiver_pos(),
            signal_matrix,
            carrier_frequency,
            offset=d.orientation)
    signal_matrixs.append(signal_matrix)
    beamformer_outputs.append(beam_former_outputs_at_t)
    assert(np.isclose(signal_matrixs[0],signal_matrixs[-1]).all())
    assert(np.isclose(beamformer_outputs[0],beamformer_outputs[-1]).all())
plt.plot(beamformer_outputs[0])
plt.show()
print("PASS!")

