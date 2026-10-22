"""Export diagnostic plots from completed rollouts (truth used only for audit)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'lerobot_policy/outputs/existing398_20261021';REPORT=ROOT/'reports/existing398_quality_20261021'
# Same initial scene/seed, explicitly a diagnostic regression comparison.
paths=[('Original frozen policy',OUT/'final_test/index_000_seed_01/simulation/rollout.npz'),
       ('Monotonic phase memory',OUT/'latched_regression/index_005_seed_01/simulation/rollout.npz'),
       ('Valid action constraints (N=1)',OUT/'valid_actions_probe_n1/index_005_seed_01/simulation/rollout.npz')]
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for label,path in paths:
    with np.load(path) as z:
        distance=np.linalg.norm(z['cube_position_world'][:,0,:2]-z['cube_position_world'][:,1,:2],axis=1)*1000
        axes[0].plot(np.arange(40),distance,label=label)
        axes[1].plot(np.arange(24,40),distance[24:],label=label)
for ax in axes:
    ax.axhline(5,color='black',linestyle='--',linewidth=1,label='5 mm criterion')
    ax.set_xlabel('Control frame');ax.set_ylabel('Cube A-B XY distance (mm)');ax.grid(alpha=.2)
axes[0].set_title('Whole rollout');axes[1].set_title('Placement stage');axes[1].set_ylim(0,30)
axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(REPORT/'phase_memory_diagnostic.png',dpi=180);plt.close(fig)

from rgbd_goal_geometry_v2 import view_geometry,GoalMemory
p=OUT/'episode_gate_wrist_remaining_closed/index_002_seed_00/simulation'
with np.load(p/'rollout.npz') as z:
    rgb=[z['camera_1_rgb'][0],z['camera_2_rgb'][0]];dep=z['depth'][0];k=z['camera_intrinsics'];poses=z['camera_to_world'];base=z['base_to_world']
    views=[view_geometry(r,d,ki,t,base) for r,d,ki,t in zip(rgb,dep,k,poses)]
    points=np.concatenate([v['points'][v['green']] for v in views]);normals=np.concatenate([v['normals'][v['green']] for v in views])
    m=GoalMemory();new=m.observe(rgb,dep,k,poses,base)[:3]
d=json.loads((p/'result.json').read_text());old=np.asarray(d['goal_memory_audit']['measured_initial_goal_base'])
truth=(np.linalg.inv(base)@np.r_[d['initial_cube_states'][1]['position'],1])[:3]
selected=np.all(np.abs(points[:,:2]-new[:2])<.055,axis=1)
fig,ax=plt.subplots(figsize=(6.5,5.5))
ax.scatter(points[selected,0],points[selected,1],s=5,color='#7f9f7d',alpha=.5,label='Measured green RGB-D points')
for centre,color,label in [(old,'#bd4b76','Old estimate (12 mm error)'),(new,'#2368b8','Surface-normal estimate')]:
    ax.add_patch(Rectangle(centre[:2]-.03,.06,.06,fill=False,edgecolor=color,linewidth=1.6))
    ax.scatter(*centre[:2],s=65,color=color,marker='x',label=label)
ax.scatter(*truth[:2],s=80,color='black',marker='+',label='Simulator truth: audit only')
ax.set_aspect('equal');ax.set_xlabel('FR3 base X (m)');ax.set_ylabel('FR3 base Y (m)');ax.set_title('Target geometry under partial visibility')
ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(REPORT/'goal_geometry_diagnostic.png',dpi=180);plt.close(fig)
print('Exported diagnostic figures')
