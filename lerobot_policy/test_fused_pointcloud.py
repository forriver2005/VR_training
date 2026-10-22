"""Fused-cloud geometry, view coverage, determinism and legacy compatibility."""
import unittest
import numpy as np
from delivery_policy_contract import ROOT,load_entries,wrist_point_cloud,fused_point_cloud,fused_view_quotas

class FusedCloudTest(unittest.TestCase):
    def setUp(self):
        self.row=load_entries(ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021/pilot_scene_checks_4hz.jsonl')[-1]

    def test_each_sample_reprojects_to_its_measured_camera(self):
        with np.load(ROOT/self.row['npz']) as z:
            for t in [0,16,32]:
                args=(z['depth'][t],z['camera_intrinsics'],z['camera_to_world'],z['wrist_depth'][t],
                      z['wrist_camera_intrinsics'],z['wrist_camera_to_world'][t],z['base_to_world'])
                cloud,counts=fused_point_cloud(*args)
                self.assertEqual(cloud.shape,(1024,3));self.assertTrue((counts>0).all())
                np.testing.assert_array_equal(cloud,fused_point_cloud(*args)[0])
                views=[(slice(0,256),args[0][0],args[1][0],args[2][0]),
                       (slice(256,512),args[0][1],args[1][1],args[2][1]),
                       (slice(512,1024),args[3],args[4],args[5])]
                for selected,depth,k,pose in views:
                    transform=np.linalg.inv(pose)@args[6]
                    cam=cloud[selected]@transform[:3,:3].T+transform[:3,3]
                    u=cam[:,0]/cam[:,2]*k[0,0]+k[0,2];v=cam[:,1]/cam[:,2]*k[1,1]+k[1,2]
                    ui=np.floor(u).astype(int);vi=np.floor(v).astype(int)
                    self.assertTrue(((ui>=0)&(ui<256)&(vi>=0)&(vi<256)).all())
                    np.testing.assert_allclose(cam[:,2],depth[vi,ui],rtol=0,atol=1e-6)
                    np.testing.assert_allclose(u-ui,.5,rtol=0,atol=1e-3)
                    np.testing.assert_allclose(v-vi,.5,rtol=0,atol=1e-3)

    def test_legacy_default_point_count_remains_1024(self):
        with np.load(ROOT/self.row['npz']) as z:
            args=(z['wrist_depth'][0],z['wrist_camera_intrinsics'],z['wrist_camera_to_world'][0],z['base_to_world'])
            np.testing.assert_array_equal(wrist_point_cloud(*args)[0],wrist_point_cloud(*args,count=1024)[0])
            self.assertEqual(wrist_point_cloud(*args)[0].shape,(1024,3))
            with self.assertRaises(ValueError):wrist_point_cloud(np.zeros((256,256)),*args[1:])

    def test_missing_view_uses_only_remaining_measured_points(self):
        with np.load(ROOT/self.row['npz']) as z:
            d=z['depth'][0].copy();d[0]=0
            cloud,counts=fused_point_cloud(d,z['camera_intrinsics'],z['camera_to_world'],z['wrist_depth'][0],
                z['wrist_camera_intrinsics'],z['wrist_camera_to_world'][0],z['base_to_world'])
            np.testing.assert_array_equal(fused_view_quotas(counts),[0,512,512])
            expected=np.concatenate([wrist_point_cloud(d[1],z['camera_intrinsics'][1],z['camera_to_world'][1],z['base_to_world'],512)[0],
                wrist_point_cloud(z['wrist_depth'][0],z['wrist_camera_intrinsics'],z['wrist_camera_to_world'][0],z['base_to_world'],512)[0]])
            np.testing.assert_array_equal(cloud,expected)
            with self.assertRaises(ValueError):
                fused_point_cloud(np.zeros_like(d),z['camera_intrinsics'],z['camera_to_world'],np.zeros((256,256)),
                    z['wrist_camera_intrinsics'],z['wrist_camera_to_world'][0],z['base_to_world'])

if __name__=='__main__':unittest.main()
