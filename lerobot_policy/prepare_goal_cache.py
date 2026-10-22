import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from delivery_policy_contract import load_entries
from rgbd_object_features import load_goal_features

def work(e):
    f=load_goal_features(e)
    return dict(uid=e['uid'],first_valid=bool(f[0,6]),valid_frames=int((f[:,6]>0).sum()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--list',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(work,e) for e in load_entries(a.list)]
        for f in as_completed(fs):
            rows.append(f.result())
            if len(rows)%25==0:print('GOAL',len(rows),'/',len(fs),flush=True)
    a.output.write_text(json.dumps(rows,indent=2)+'\n');print('COMPLETE',len(rows),'initially valid',sum(r['first_valid'] for r in rows),flush=True)

if __name__=='__main__':main()
