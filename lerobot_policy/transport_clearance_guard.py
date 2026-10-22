"""Keep learned transport above the measured destination approach height.

Uses robot/goal observations and the learned transport event, not obstacle poses
or object ground truth. It preserves the policy's horizontal waypoint direction.
"""
import numpy as np
DEFAULT_TRANSPORT_GUARD=dict(hover_offset_m=.20,height_tolerance_m=.03,xy_tolerance_m=.006,
    max_horizontal_step_m=.08,max_rise_step_m=.10,max_descent_step_m=.08)

class TransportClearanceGuard:
    def __init__(self,config,stack_height):self.config=dict(config);self.stack_height=float(stack_height);self.reset()
    def reset(self):
        self.arrived=None;self.lifting=None;self.hold_xy=None;self.override_count=0
    def apply(self,proposed,state,goal,started,release_sent):
        cmd=np.asarray(proposed,dtype=np.float32).copy();old=cmd.copy();state=np.asarray(state);goal=np.asarray(goal)
        b=len(cmd);cfg=self.config
        if self.arrived is None:
            self.arrived=np.zeros(b,dtype=bool);self.lifting=np.zeros(b,dtype=bool);self.hold_xy=np.zeros((b,2),dtype=np.float32)
        if len(self.arrived)!=b:raise ValueError('Reset transport guard before changing batch size')
        started=np.asarray(started,dtype=bool).reshape(b);released=np.asarray(release_sent,dtype=bool).reshape(b)
        clearance=goal[:,2]+cfg['hover_offset_m']
        for i in range(b):
            if goal[i,6]<=0 or not started[i] or released[i]:continue
            near=np.linalg.norm(state[i,:2]-goal[i,:2])<=cfg['xy_tolerance_m']
            high=state[i,2]>=clearance[i]-cfg['height_tolerance_m']
            self.arrived[i] |= bool(near and high)
            if not self.arrived[i]:
                if not high:
                    if not self.lifting[i]:self.hold_xy[i]=state[i,:2]
                    self.lifting[i]=True;cmd[i,:2]=self.hold_xy[i]
                    cmd[i,2]=min(max(float(cmd[i,2]),float(clearance[i])),float(state[i,2])+cfg['max_rise_step_m'])
                else:
                    self.lifting[i]=False
                    delta=cmd[i,:2]-state[i,:2];distance=float(np.linalg.norm(delta))
                    if distance>cfg['max_horizontal_step_m']:cmd[i,:2]=state[i,:2]+delta*(cfg['max_horizontal_step_m']/distance)
                    cmd[i,2]=min(max(float(cmd[i,2]),float(clearance[i])),float(state[i,2])+cfg['max_rise_step_m'])
            else:
                self.lifting[i]=False;cmd[i,:2]=goal[i,:2]
                floor=goal[i,2]+self.stack_height
                cmd[i,2]=max(float(cmd[i,2]),float(floor),float(state[i,2])-cfg['max_descent_step_m'])
                cmd[i,2]=min(float(cmd[i,2]),float(state[i,2])+cfg['max_rise_step_m'])
        changed=np.any(np.abs(cmd-old)>1e-6,axis=1);self.override_count+=int(changed.sum())
        return cmd,dict(transport_guard_changed=changed,transport_goal_column_reached=self.arrived.copy(),transport_waiting_for_height=self.lifting.copy(),
            transport_clearance_height=clearance.copy(),transport_guard_override_count=self.override_count)
