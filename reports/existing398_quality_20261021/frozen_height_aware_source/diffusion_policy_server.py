#!/usr/bin/env python3
"""Serve EMA RGB diffusion using the same inference as offline reconstruction."""
import argparse
import socket
from pathlib import Path
import numpy as np
import torch
from diffusion_overfit_common import CAMERAS, Inference
from delivery_policy_contract import sha256
from socket_protocol import read_command, receive_arrays, send_arrays, command_message

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--port',type=int,default=5560)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--listen-fd',type=int,help='inherit an already-bound listener reserved by the launcher')
    p.add_argument('--ready-token',default='')
    args=p.parse_args()
    torch.set_num_threads(4)
    model=Inference(args.checkpoint,args.device,args.seed)
    contract_hash=sha256(args.checkpoint/'run_config.json')
    with (socket.socket(fileno=args.listen_fd) if args.listen_fd is not None else socket.socket()) as listener:
        if args.listen_fd is None:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            listener.bind((args.host,args.port));listener.listen(1)
        print(f'Listening on {args.host}:{args.port}; EMA H=2 K=16 A={model.action_steps} DDPM100',flush=True)
        while True:
            conn,address=listener.accept()
            episode_id=None; previous_request=-1
            try:
                with conn:
                    while True:
                        request=receive_arrays(conn); command=read_command(request)
                        if command=='ping':send_arrays(conn,{**command_message('pong'),'ready_token':np.asarray(args.ready_token),'checkpoint_contract_sha256':np.asarray(contract_hash)})
                        elif command=='reset':
                            model.reset(int(request.get('seed',np.asarray(args.seed)).item()))
                            episode_id=int(request.get('episode_id',np.asarray(0)).item())
                            previous_request=-1
                            send_arrays(conn,command_message('ok'))
                        elif command=='predict':
                            current_episode=int(request.get('episode_id',np.asarray(0)).item())
                            request_id=int(request.get('request_id',np.asarray(previous_request+1)).item())
                            if current_episode!=episode_id or request_id!=previous_request+1:
                                raise ValueError('Predict requires reset and consecutive episode-local requests')
                            action,diag=model.predict(request['observation.state'],[request[k] for k in CAMERAS],
                                request.get('observation.point_cloud'),request.get('observation.elapsed_same_target_s'),request.get('observation.goal_features'))
                            previous_request=request_id
                            send_arrays(conn,{'command':np.asarray('action'),'action':action,
                                'episode_id':np.asarray(episode_id),'request_id':np.asarray(request_id),
                                **{k:np.asarray(v) for k,v in diag.items()}})
                        else:raise ValueError(f'Unknown command {command}')
            except (ConnectionError,OSError) as exc:
                print(f'Client disconnected: {exc}',flush=True)

if __name__=='__main__':main()
