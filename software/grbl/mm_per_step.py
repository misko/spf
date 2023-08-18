teeth_per_revolution=20
spacing_per_tooth=2 # 2mm
gear_ratio=26+103.0/121
degrees_per_step_wo_gear=1.8

degrees_per_step=(degrees_per_step_wo_gear/gear_ratio)
steps_per_revolution=360/degrees_per_step
micro_stepping=16
steps_per_mm=micro_stepping*steps_per_revolution/(teeth_per_revolution*spacing_per_tooth)



#$100=2148.000
#$101=2148.000
#$102=800.000
#$103=800.000
#$110=700.000
#$111=700.000

print('$100=%d\n\
$101=%d\n\
$110=1000\n\
$111=1000\n\
$120=50\n\
$121=50\n' % (int(steps_per_mm),int(steps_per_mm)))
