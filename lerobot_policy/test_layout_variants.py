import unittest
import numpy as np
from layout_variants import shifted_episode

class LayoutTests(unittest.TestCase):
    def test_pair_keeps_grasp_and_reset_but_moves_place(self):
        e=dict(uid='source',source_npz_sha256='hash')
        base=np.eye(4);base[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
        data=dict(action=np.tile([.5,0,.2,1,0,0,0,.04],(40,1)),reset_target=np.array([.5,0,.3,1,0,0,0,.04]),base_to_world=base)
        diag=dict(initial_cube_states=[dict(position=[0,0,.02]) for _ in range(2)])
        spec=dict(source_uid='source',source_npz_sha256='hash',a_delta_base_m=[0,0,0],b_delta_base_m=[0,.025,0])
        d,z=shifted_episode(e,data,diag,spec)
        np.testing.assert_array_equal(d['action'][:20],data['action'][:20])
        np.testing.assert_array_equal(d['reset_target'],data['reset_target'])
        np.testing.assert_allclose(d['action'][26:,1],.025)
        np.testing.assert_array_equal(d['action'][:,3:],data['action'][:,3:])
        np.testing.assert_allclose(z['initial_cube_states'][1]['position'],[-.025,0,.02])
        np.testing.assert_allclose(data['action'][:,1],0)

if __name__=='__main__':unittest.main()
