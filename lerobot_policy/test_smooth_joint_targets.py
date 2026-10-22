import unittest
import numpy as np
from smooth_joint_targets import TargetFilter

class FilterTest(unittest.TestCase):
    def test_reference_converges_without_overshoot_and_stays_at_target(self):
        q0=np.zeros(7);target=np.array([.1,-.2,.3,-.4,.5,-.6,.7])
        f=TargetFilter(q0,.04,1/120,.04,.04);f.target(target,.016)
        history=np.stack([f.advance()[0] for _ in range(30)])
        np.testing.assert_allclose(history[-1],target*(1-np.exp(-.25/.04)),atol=1e-15)
        self.assertTrue(np.all(np.abs(history)<=np.abs(target)))
        self.assertTrue(np.all(np.diff(np.abs(history),axis=0)>=0))
        self.assertLess(f.max_joint_step,np.max(np.abs(target))*.2)
        for _ in range(120):f.advance()
        np.testing.assert_allclose(f.q,target,atol=1e-12);self.assertAlmostEqual(f.grip,.016,places=12)

if __name__=='__main__':unittest.main()
