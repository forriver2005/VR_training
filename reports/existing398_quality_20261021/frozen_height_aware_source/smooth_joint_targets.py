"""Optional inner-loop reference smoothing, with unchanged static drive gains."""
import math
import numpy as np

class TargetFilter:
    def __init__(self,q,grip,physics_dt,joint_tau,gripper_tau):
        self.q=np.asarray(q,dtype=np.float64).copy();self.grip=float(grip)
        self.q_goal=self.q.copy();self.grip_goal=self.grip
        self.dt=float(physics_dt)
        self.aq=1. if joint_tau==0 else -math.expm1(-self.dt/joint_tau)
        self.ag=1. if gripper_tau==0 else -math.expm1(-self.dt/gripper_tau)
        self.max_joint_step=0.;self.max_grip_step=0.;self.steps=0
    def target(self,q,grip):
        self.q_goal=np.asarray(q,dtype=np.float64).copy();self.grip_goal=float(grip)
    def advance(self):
        previous=self.q.copy();old_grip=self.grip
        self.q+=self.aq*(self.q_goal-self.q);self.grip+=self.ag*(self.grip_goal-self.grip)
        self.max_joint_step=max(self.max_joint_step,float(np.max(np.abs(self.q-previous))))
        self.max_grip_step=max(self.max_grip_step,abs(self.grip-old_grip));self.steps+=1
        return self.q.copy(),self.grip


def attach_target_filter(runtime,config):
    """Install only after the original, frozen reset has finished."""
    dt=float(runtime.sim.get_physics_dt())
    if abs(dt-config['physics_dt_s'])>1e-10:raise ValueError('Controller physics rate mismatch')
    if runtime.q_target is None:raise ValueError('Initialize/reset the robot before filtering targets')
    control=TargetFilter(runtime.q_target,runtime.grip_target,dt,config['joint_tau_s'],config['gripper_tau_s'])
    original_command=runtime.command
    def command(q,grip):control.target(q,grip)
    def step(n):
        start=runtime.sim.current_time
        if not runtime.sim.is_playing():raise RuntimeError('Physics timeline is not playing')
        for _ in range(n):
            q,g=control.advance();original_command(q,g)
            runtime.sim.step(render=False,update_fabric=True)
        if abs(runtime.sim.current_time-start-n*dt)>2e-6:raise RuntimeError('Filtered controller changed physics timing')
    runtime.command=command;runtime.step=step
    return control
