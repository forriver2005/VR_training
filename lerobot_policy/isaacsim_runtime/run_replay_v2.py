"""Run independent replays with bounded deferred fresh-process recovery."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def retryable_log(text):
    # A bounded fresh-process retry also covers Kit/RTX startup and teardown
    # crashes. The caller still stops after max_restarts to avoid a crash loop.
    if '[Fatal]' in text:
        return True
    return any(marker in text for marker in (
        'RGB_CAPTURE_FAILED', 'RGB render did not become valid',
        'REPLAY_INITIAL_STATE_MISMATCH'))

def retryable_result(result):
    """Return true for completed replays whose sensor evidence is transient."""
    reasons=set(result.get('failure_reasons') or [])
    return bool(reasons & {
        'replay_rgb_projection', 'camera_calibration_audit',
        'replay_blank_rgb', 'blank_rgb_camera_1', 'blank_rgb_camera_2',
    })


def replay_once(isaac_python,script,dataset,episode,report_dir=None,headless=True):
    """Run one Isaac replay process and return its bounded result."""
    dataset=Path(dataset).resolve()
    folder=Path(report_dir).resolve() if report_dir else dataset/'replays'
    logs=folder/'attempts'
    logs.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_path=logs/f'{episode}_{stamp}.log'
    command=[str(isaac_python),'-u',str(script),str(dataset),'--episode',episode]
    if report_dir:command+=['--report-dir',str(folder)]
    if not headless:command+=['--no-headless']
    print(f'ISAAC_REPLAY_PROCESS: {episode}; log: {log_path}',flush=True)
    lines=[]
    with log_path.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            stream.write(line);stream.flush();print(line,end='',flush=True);lines.append(line)
        process.stdout.close()
        exit_code=process.wait()
    text=''.join(lines)
    completed=exit_code==0 and 'REPLAY_RESULT ' in text and '[Fatal]' not in text
    result_path=folder/f'{episode}_verification.json'
    result=json.loads(result_path.read_text(encoding='utf-8')) if completed and result_path.is_file() else {}
    completed=bool(completed and result and (folder/f'{episode}.gif').is_file())
    passed=bool(completed and result.get('success') and not result.get('failure_reasons'))
    retryable=bool(completed and retryable_result(result)) or (not completed and retryable_log(text))
    return dict(exit_code=exit_code,completed=completed,passed=passed,retryable=retryable,
                log=str(log_path.relative_to(folder)))


def run_replay(isaac_python,script,dataset,episode,max_restarts=2,report_dir=None,headless=True):
    """Backward-compatible one-episode wrapper around a deferred replay pass."""
    passed,attempts=deferred_replay(isaac_python,script,dataset,[episode],max_restarts,report_dir,headless)
    return 0 if passed else (2 if attempts and attempts[-1].get('completed') else 1)


def deferred_replay(isaac_python,script,dataset,episodes,max_restarts=2,report_dir=None,headless=True):
    """Replay all episodes first, then retry camera-transient failures later.

    Camera/RTX failures are often episode-specific. Continuing the first pass
    gives subsequent trajectories a chance to run before spending a fresh
    Isaac process on a failed episode.
    """
    dataset=Path(dataset).resolve()
    folder=Path(report_dir).resolve() if report_dir else dataset/'replays'
    (folder/'attempts').mkdir(parents=True,exist_ok=True)
    attempt_records={}
    pending=list(episodes)
    for episode in episodes:
        record=replay_once(isaac_python,script,dataset,episode,report_dir,headless)
        attempt_records[episode]=[record]
        if record['passed'] or not record['retryable']:
            pending.remove(episode)
    deferred=[episode for episode in pending]
    for episode in deferred:
        print(f'ISAAC_REPLAY_DEFERRED: retrying {episode} after the first pass.',flush=True)
        for restart in range(max_restarts):
            record=replay_once(isaac_python,script,dataset,episode,report_dir,headless)
            attempt_records[episode].append(record)
            if record['passed'] or not record['retryable'] or restart+1==max_restarts:
                break
            time.sleep(2)
    final_attempts=[];all_passed=True
    for episode in episodes:
        records=attempt_records[episode]
        final=records[-1];passed=final['passed'];all_passed=all_passed and passed
        (folder/f'{episode}_restart_report.json').write_text(
            json.dumps(dict(episode_id=episode,passed=passed,
                attempts=[dict(attempt=i+1,**row) for i,row in enumerate(records)]),indent=2),
            encoding='utf-8')
        final_attempts.extend(records)
    if not report_dir:
        # Child hashing finishes before these attempt logs/reports are final.
        sys.path.insert(0,str(dataset/'source/Tools'))
        from fr3_v2_delivery import update_checksums
        update_checksums(dataset)
    return all_passed,final_attempts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path)
    p.add_argument('--episode',required=True,action='append',
                   help='Episode id; repeat to schedule a batch with deferred retries.')
    p.add_argument('--isaac-python',required=True,type=Path)
    p.add_argument('--script',required=True,type=Path)
    p.add_argument('--max-restarts',type=int,default=2)
    p.add_argument('--report-dir',type=Path)
    p.add_argument('--no-headless',action='store_true')
    a=p.parse_args()
    if a.max_restarts<0:p.error('--max-restarts must be nonnegative')
    passed,attempts=deferred_replay(a.isaac_python,a.script,a.dataset,a.episode,a.max_restarts,
                                   a.report_dir,not a.no_headless)
    if passed:return 0
    return 2 if attempts and all(row['completed'] for row in attempts) else 1


if __name__=='__main__':
    raise SystemExit(main())
