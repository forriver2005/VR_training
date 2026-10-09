"""Focused checks for measured-depth geometry and PointNet invariance."""
import unittest
import numpy as np
import torch
from rgbd_geometry import BASE_WORLD, LOW, HIGH, backproject, sample_fps, normalize_points
from pointcloud_diffusion import PointNetEncoderXYZ


class PointCloudContractTest(unittest.TestCase):
    def test_optical_depth_and_base_transform(self):
        depth=np.array([[2.,np.nan],[0.,2.]],dtype=np.float32)
        k=np.array([[2.,0.,1.],[0.,2.,1.],[0.,0.,1.]])
        t=np.eye(4)
        t[:3,:3]=np.diag([1.,-1.,-1.])
        t[:3,3]=BASE_WORLD+[.5,0.,2.]
        points,invalid=backproject(depth,k,t)
        np.testing.assert_allclose(points,[[0.,.5,0.],[1.,-.5,0.]],atol=1e-7)
        self.assertEqual(invalid,.5)

    def test_sampling_and_empty_cloud(self):
        points=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
        sampled=sample_fps(points,5)
        self.assertEqual(len(np.unique(sampled[:3],axis=0)),3)
        np.testing.assert_array_equal(sampled,sample_fps(points,5))
        np.testing.assert_array_equal(sampled[3:],sampled[:2])
        with self.assertRaises(ValueError): sample_fps(np.empty((0,3)))

    def test_fixed_normalization(self):
        np.testing.assert_allclose(normalize_points(np.stack([LOW,HIGH])),[[-1]*3,[1]*3])

    def test_pointnet_permutation_invariance(self):
        torch.manual_seed(0)
        model=PointNetEncoderXYZ().eval()
        cloud=torch.randn(2,1024,3)
        with torch.no_grad():
            torch.testing.assert_close(model(cloud),model(cloud[:,torch.randperm(1024)]))


if __name__=='__main__': unittest.main()
