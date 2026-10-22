"""Freeze 25 validation trajectories by action-path coverage; no split migration."""
from pathlib import Path
import json
import numpy as np
from delivery_policy_contract import ROOT,load_entries,sha256

def main():
    folder=ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021'
    source=folder/'validation_4hz.jsonl';entries=load_entries(source)
    feature=[]
    for e in entries:
        with np.load(ROOT/e['npz']) as z:
            feature.append(z['action'][20:31,:3].astype(np.float64).reshape(-1))
    feature=np.asarray(feature)
    index=int(np.argmin(np.sum((feature-feature.mean(0))**2,axis=1)))
    indices=[];distance=np.full(len(entries),np.inf)
    for _ in range(min(25,len(entries))):
        indices.append(index);distance=np.minimum(distance,np.sum((feature-feature[index])**2,axis=1))
        distance[indices]=-1;index=int(np.argmax(distance))
    (folder/'validation_quick_4hz.jsonl').write_text(''.join(json.dumps(entries[i])+'\n' for i in sorted(indices)))
    (folder/'quick_validation_selection.json').write_text(json.dumps(dict(count=len(indices),source_sha256=sha256(source),
        method='deterministic farthest-point coverage of transport/place XYZ action paths, first nearest centroid',
        frame_indices=list(range(20,31)),selected_uids=[entries[i]['uid'] for i in sorted(indices)],
        membership='all 100 remain exclusively in validation; selected 25 are an evaluation subset'),indent=2)+'\n')
    print('Frozen',len(indices),'validation trajectories; all 100 retained in validation.')

if __name__=='__main__':main()
