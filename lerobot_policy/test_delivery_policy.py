"""Boundary isolation, command timer robustness and real archived depth checks."""
import unittest
import numpy as np
import torch
from delivery_policy_contract import ROOT,HoldTimer,BOUNDED_TIMER,timer_sequence,load_entries,wrist_point_cloud
from train_delivery_diffusion import make_batch

class DeliveryPolicyTest(unittest.TestCase):
    def test_windows_do_not_cross_episodes_and_execution_slice(self):
        values=torch.stack([torch.arange(40),100+torch.arange(40)]).float()[:,:,None]
        b=make_batch({'observation.state':values},values,torch.tensor([0,1,0]),torch.tensor([0,39,8]))
        self.assertEqual(b['observation.state'][:,:,0].tolist(),[[0,0],[138,139],[7,8]])
        self.assertEqual(b['action'][0,1:9,0].tolist(),list(range(8)))
        self.assertEqual(b['action'][1,:,0].tolist(),[138]+[139]*15)
        self.assertEqual(b['action'][2,1:9,0].tolist(),list(range(8,16)))
        self.assertEqual(b['observation.history_valid'][:,:,0].tolist(),[[0,1],[1,1],[1,1]])

    def test_timer_tolerates_noise_but_not_accumulating_drift(self):
        a=np.tile([.5,0,.2,1,0,0,0,.02],(6,1)).astype(float)
        noisy=a.copy();noisy[:,:3]+=np.random.default_rng(0).normal(0,.0001,(6,3))
        np.testing.assert_allclose(timer_sequence(noisy,.25),np.arange(6)*.25)
        timer=HoldTimer()
        for offset in [0,.0008,.0016]:
            target=a[0].copy();target[0]+=offset;timer.advance(target,.25)
        self.assertEqual(timer.elapsed,.25)
        flipped=a.copy();flipped[1::2,3:7]*=-1
        np.testing.assert_allclose(timer_sequence(flipped,.25),np.arange(6)*.25)

    def test_bounded_timer_does_not_accumulate_unlimited_target_drift(self):
        timer=HoldTimer(BOUNDED_TIMER)
        for offset in [0,.002,.004,.006]:
            timer.advance([.5+offset,0,.2,1,0,0,0,.02],.25)
        self.assertEqual(timer.elapsed,.25)
        timer=HoldTimer(BOUNDED_TIMER)
        timer.advance([.5,0,.2,1,0,0,0,.02],.25)
        timer.advance([.51,0,.2,1,0,0,0,.02],.25)
        self.assertEqual(timer.elapsed,.25)

    def test_all_pilot_clouds_are_measured_depth_points(self):
        row=load_entries(ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021/overfit_4hz.jsonl')[0]
        with np.load(ROOT/row['npz']) as z:
            for t in range(40):
                cloud,count=wrist_point_cloud(z['wrist_depth'][t],z['wrist_camera_intrinsics'],z['wrist_camera_to_world'][t],z['base_to_world'])
                self.assertGreater(count,0)
                to_camera=np.linalg.inv(z['wrist_camera_to_world'][t])@z['base_to_world']
                cam=cloud@to_camera[:3,:3].T+to_camera[:3,3];k=z['wrist_camera_intrinsics']
                u=cam[:,0]/cam[:,2]*k[0,0]+k[0,2];v=cam[:,1]/cam[:,2]*k[1,1]+k[1,2]
                ui=np.floor(u).astype(int);vi=np.floor(v).astype(int)
                np.testing.assert_allclose(cam[:,2],z['wrist_depth'][t,vi,ui],atol=1e-6,rtol=0)
                np.testing.assert_allclose(u-ui,.5,atol=1e-3,rtol=0)
                np.testing.assert_allclose(v-vi,.5,atol=1e-3,rtol=0)

if __name__=='__main__':unittest.main()
