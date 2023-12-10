from spf.grbl.grbl_interactive import GRBLManager, home_pA, home_pB

import numpy as np


def test_steps_and_steps_inverse():
  g=GRBLManager(None,
    pA=home_pA,
    pB=home_pB,
    motor_mapping=[
      ['X','Y'],
      ['Z','A']
    ],
    bounding_box=[])

  n=10
  for x in np.linspace(0,2000,n+1):
    for y in np.linspace(0,4000,n+1):
      p=np.array([x,y])
      steps=g.to_steps(np.array(p))
      back=g.from_steps(*steps)
      assert(np.isclose(p,back).all())

def test_xaxis():
  g=GRBLManager(None,
    pA=home_pA,
    pB=home_pB,
    motor_mapping=[
      ['X','Y'],
      ['Z','A']
    ],
    bounding_box=[])

  n=10
  y=0
  for x in np.linspace(-2000,2000,n+1):
    p=np.array([x,y])
    steps=g.to_steps(np.array(p))
    back=g.from_steps(*steps)
    if x>0:
      assert(steps[0]==-steps[1]) # if we are sliding across the x axis we should have a constant length , so decrease in one is increase in the other
    else:
      assert(steps[0]==steps[1])  # not physically possible
    assert(steps[0]==-x)
    assert(np.isclose(p,back).all())


