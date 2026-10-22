"""Freeze five diverse training paths per scene without using validation scores."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from delivery_policy_contract import ROOT,load_entries,sha256

def main():
    folder=ROOT/'data/training_manifests/fr3v2_cube40_cube60_20261021'
    source=folder/'train_4hz.jsonl';groups=defaultdict(list)
    for e in load_entries(source):groups[e['scene_id']].append(e)
    selected=[];representatives=[]
    for scene,entries in groups.items():
        features=[]
        for e in entries:
            with np.load(ROOT/e['npz']) as z:features.append(z['action'][20:31,:3].astype(float).reshape(-1))
        features=np.asarray(features);dist=np.full(len(entries),np.inf)
        index=int(np.argmin(np.sum((features-features.mean(0))**2,axis=1)));indices=[]
        for _ in range(5):
            indices.append(index);dist=np.minimum(dist,np.sum((features-features[index])**2,axis=1))
            dist[indices]=-1;index=int(np.argmax(dist))
        selected.extend(entries[i] for i in sorted(indices));representatives.append(entries[indices[0]])
    validation=load_entries(folder/'validation_quick_4hz.jsonl')
    representatives.append(validation[0])
    for name,entries in [('pilot20_4hz',selected),('pilot_scene_checks_4hz',representatives)]:
        (folder/f'{name}.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in entries))
    record=dict(training_count=len(selected),training_scenes=len(groups),training_list_sha256=sha256(source),
        selection='centroid-nearest start then farthest action-path coverage over frames20:31, five per training scene',
        representatives=[dict(index=i,uid=e['uid'],role='training' if i<4 else 'validation') for i,e in enumerate(representatives)],
        test_data_used=False)
    (folder/'pilot20_selection.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
