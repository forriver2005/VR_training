#!/usr/bin/env python3
"""Serve EMA RGB diffusion using the same inference as offline reconstruction."""
import argparse
import socket
from pathlib import Path
import numpy as np
import torch
from diffusion_overfit_common import CAMERAS, Inference
from socket_protocol import read_command, receive_arrays, send_arrays, command_message

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--port',type=int,default=5560)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    torch.set_num_threads(4)
    model=Inference(args.checkpoint,args.device,args.seed)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind((args.host,args.port));listener.listen(1)
        print(f'Listening on {args.host}:{args.port}; EMA H=2 K=16 A=8 DDPM100',flush=True)
        while True:
            conn,address=listener.accept()
            episode_id=None; previous_request=-1
            try:
                with conn:
                    while True:
                        request=receive_arrays(conn); command=read_command(request)
                        if command=='ping':send_arrays(conn,command_message('pong'))
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
                                request.get('observation.point_cloud'))
                            previous_request=request_id
                            send_arrays(conn,{'command':np.asarray('action'),'action':action,
                                'episode_id':np.asarray(episode_id),'request_id':np.asarray(request_id),
                                **{k:np.asarray(v) for k,v in diag.items()}})
                        else:raise ValueError(f'Unknown command {command}')
            except (ConnectionError,OSError) as exc:
                print(f'Client disconnected: {exc}',flush=True)

if __name__=='__main__':main()
