import unittest
import numpy as np
from release_guard import ReleaseGuard,DEFAULT_RELEASE_GUARD

class ReleaseGuardTest(unittest.TestCase):
    def test_requested_release_waits_for_nominal_command_and_measured_settling(self):
        g=ReleaseGuard(DEFAULT_RELEASE_GUARD,.05)
        goal=np.array([[.5,.1,.045,.06,.06,.06,1,1]],np.float32)
        state=np.array([[.5,.1,.145,1,0,0,0,.02]],np.float32)
        closed=state.copy();closed[:,7]=.016
        g.apply(closed,state,goal,[True],[1.])
        requested=closed.copy();requested[:,7]=.04
        out,diag=g.apply(requested,state,goal,[True],[1.])
        self.assertAlmostEqual(float(out[0,7]),.016,places=6);self.assertAlmostEqual(float(out[0,2]),.095,places=6)
        state[:,:3]=out[:,:3]
        out,_=g.apply(requested,state,goal,[True],[1.]);self.assertAlmostEqual(float(out[0,7]),.016,places=6)
        out,_=g.apply(requested,state,goal,[True],[1.]);self.assertAlmostEqual(float(out[0,7]),.016,places=6)
        out,diag=g.apply(requested,state,goal,[True],[1.]);self.assertAlmostEqual(float(out[0,7]),.04,places=6);self.assertTrue(diag['release_opening_sent'][0])
        retreat=requested.copy();retreat[:,2]=.245
        for _ in range(2):
            out,_=g.apply(retreat,state,goal,[True],[1.]);self.assertAlmostEqual(float(out[0,2]),.095,places=6)
        out,_=g.apply(retreat,state,goal,[True],[1.]);self.assertAlmostEqual(float(out[0,2]),.245,places=6)
        g.reset();initial=requested.copy()
        out,diag=g.apply(initial,state,goal,[False],[0.]);self.assertFalse(diag['release_opening_sent'][0]);np.testing.assert_array_equal(out,initial)
    def test_pose_alone_never_invents_a_release(self):
        g=ReleaseGuard(DEFAULT_RELEASE_GUARD,.05);goal=np.array([[.5,.1,.045,.06,.06,.06,1,1]],np.float32)
        state=np.array([[.5,.1,.095,1,0,0,0,.02]],np.float32);act=state.copy();act[:,7]=.016
        for _ in range(10):
            out,diag=g.apply(act,state,goal,[True],[1.]);self.assertFalse(diag['release_opening_sent'][0]);self.assertAlmostEqual(float(out[0,7]),.016,places=6)

if __name__=='__main__':unittest.main()
