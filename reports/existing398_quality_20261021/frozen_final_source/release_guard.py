"""Execute a learned release request only after a stable placement command.

Inputs are measured robot state, the RGB-D goal and the policy's own commands.
This module does not inspect object poses, contacts, rewards or expert phases.
"""
import numpy as np
DEFAULT_RELEASE_GUARD=dict(stable_intervals=2,post_open_hold_steps=2,xy_tolerance_m=.006,z_tolerance_m=.006,
    command_tolerance_m=.001,motion_tolerance_m=.0025,max_descent_per_step_m=.05,
    closed_target_m=.016,open_target_m=.04,control_dt_s=.25)

class ReleaseGuard:
    def __init__(self,config,height_offset):
        self.config=dict(config);self.height=float(height_offset);self.reset()
    def reset(self):
        self.previous_state=None;self.previous_action=None;self.has_closed=None
        self.pending=None;self.opening_sent=None;self.stable=None;self.post_hold=None
        self.override_count=0
    def apply(self,proposed,state,goal,transport_started,anchor_weight):
        command=np.asarray(proposed,dtype=np.float32).copy();before=command.copy()
        state=np.asarray(state);goal=np.asarray(goal);b=len(command);cfg=self.config
        if self.has_closed is None:
            self.has_closed=np.zeros(b,dtype=bool);self.pending=np.zeros(b,dtype=bool);self.opening_sent=np.zeros(b,dtype=bool)
            self.stable=np.zeros(b,dtype=np.int64);self.post_hold=np.zeros(b,dtype=np.int64)
        if len(self.has_closed)!=b:raise ValueError('Reset release guard before changing batch size')
        valid=goal[:,6]>0;place=goal[:,:3].copy();place[:,2]+=self.height
        self.has_closed |= command[:,7]<.028
        wants_open=command[:,7]>=.035
        started=np.asarray(transport_started,dtype=bool).reshape(b)
        arrived_intent=np.asarray(anchor_weight).reshape(b)>=.95
        self.pending |= self.has_closed&started&valid&wants_open&arrived_intent&~self.opening_sent
        for i in range(b):
            if not valid[i]:
                if self.has_closed[i] and started[i] and not self.opening_sent[i]:command[i,7]=cfg['closed_target_m']
                continue
            xy=float(np.linalg.norm(state[i,:2]-place[i,:2]))
            near=xy<=cfg['xy_tolerance_m'] and abs(float(state[i,2]-place[i,2]))<=cfg['z_tolerance_m']
            settled=False
            if self.previous_action is not None and self.previous_state is not None:
                nominal=np.linalg.norm(self.previous_action[i,:3]-place[i])<=cfg['command_tolerance_m']
                still=np.linalg.norm(state[i,:3]-self.previous_state[i,:3])<=cfg['motion_tolerance_m']
                settled=bool(nominal and still and near and self.previous_action[i,7]<.028)
            self.stable[i]=self.stable[i]+1 if settled else 0
            if self.pending[i] and not self.opening_sent[i]:
                if self.stable[i]>=cfg['stable_intervals']:
                    self.opening_sent[i]=True;self.pending[i]=False
                    self.post_hold[i]=cfg['post_open_hold_steps']+1
                else:
                    command[i,7]=cfg['closed_target_m']
                    if xy<=cfg['xy_tolerance_m']:
                        command[i,:2]=place[i,:2]
                        command[i,2]=max(float(place[i,2]),float(state[i,2])-cfg['max_descent_per_step_m'])
                    elif arrived_intent[i]:
                        # Complete the policy's requested horizontal arrival
                        # while holding height; do not descend diagonally early.
                        command[i,:2]=place[i,:2]
                        command[i,2]=max(float(command[i,2]),float(state[i,2]),float(place[i,2]))
            elif self.has_closed[i] and started[i] and wants_open[i] and not arrived_intent[i] and not self.opening_sent[i]:
                command[i,7]=cfg['closed_target_m']
            if self.opening_sent[i]:
                command[i,7]=cfg['open_target_m'];command[i,:2]=place[i,:2]
                if self.post_hold[i]>0:
                    command[i,2]=place[i,2];self.post_hold[i]-=1
                else:command[i,2]=max(float(command[i,2]),float(place[i,2]))
        changed=np.any(np.abs(command-before)>1e-6,axis=1);self.override_count+=int(changed.sum())
        self.previous_action=command.copy();self.previous_state=state.copy()
        return command,dict(release_guard_changed=changed,release_pending=self.pending.copy(),release_opening_sent=self.opening_sent.copy(),release_stable_intervals=self.stable.copy(),release_guard_override_count=self.override_count)
