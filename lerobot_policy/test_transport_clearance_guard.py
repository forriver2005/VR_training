import unittest
import numpy as np
from transport_clearance_guard import TransportClearanceGuard,DEFAULT_TRANSPORT_GUARD

class TransportGuardTest(unittest.TestCase):
    def test_raise_before_horizontal_motion_for_elevated_destination(self):
        g=TransportClearanceGuard(DEFAULT_TRANSPORT_GUARD,.05)
        goal=np.array([[.6,.1,.26,.06,.06,.06,1,1]],np.float32)
        state=np.array([[.49,-.18,.235,1,0,0,0,.02]],np.float32)
        desired=np.array([[.55,-.01,.245,1,0,0,0,.016]],np.float32)
        untouched,_=g.apply(desired,state,goal,[False],[False]);np.testing.assert_array_equal(untouched,desired)
        cmd,info=g.apply(desired,state,goal,[True],[False]);np.testing.assert_allclose(cmd[0,:2],state[0,:2]);self.assertAlmostEqual(float(cmd[0,2]),.335,places=6);self.assertTrue(info['transport_waiting_for_height'][0])
        state[0,2]=.44
        cmd,_=g.apply(desired,state,goal,[True],[False]);self.assertLessEqual(float(np.linalg.norm(cmd[0,:2]-state[0,:2])),.080001);self.assertGreaterEqual(float(cmd[0,2]),.459999)
        state[0,:3]=[.6,.1,.46];desired[0,:3]=[.59,.09,.31]
        cmd,info=g.apply(desired,state,goal,[True],[False]);self.assertTrue(info['transport_goal_column_reached'][0]);np.testing.assert_allclose(cmd[0,:2],goal[0,:2]);self.assertAlmostEqual(float(cmd[0,2]),.38,places=6)
        g.reset();self.assertIsNone(g.arrived)

if __name__=='__main__':unittest.main()
