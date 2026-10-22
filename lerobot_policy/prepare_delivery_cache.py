"""Parallel measured-depth preprocessing; source episode data remains immutable."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from delivery_policy_contract import load_entries,load_episode,BOUNDED_TIMER,sha256

def prepare(job):
    entry,source=job;d=load_episode(entry,BOUNDED_TIMER,source)
    return dict(uid=entry['uid'],source=source,path=d['derived_point_cloud_path'],sha256=sha256(d['derived_point_cloud_path']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--list',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sources',nargs='+',default=['wrist_depth','fused_depth']);p.add_argument('--workers',type=int,default=8);args=p.parse_args()
    entries=load_entries(args.list);rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(prepare,(e,s)) for e in entries for s in args.sources]
        for f in as_completed(futures):
            rows.append(f.result())
            if len(rows)%20==0:print('CACHE',len(rows),'/',len(futures),flush=True)
    args.output.write_text(json.dumps(sorted(rows,key=lambda r:(r['source'],r['uid'])),indent=2)+'\n')

if __name__=='__main__':main()
