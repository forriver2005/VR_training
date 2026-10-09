#!/usr/bin/env bash
set -u

PLAYLIST="/home/houshengyuan/DexResearch/DemoGrasp/vnc_replay_playlist.txt"
while true; do
  DISPLAY=:3 XAUTHORITY=/home/houshengyuan/.Xauthority \
    ffplay -hide_banner -loglevel warning -nostats -autoexit -fs \
      -window_title "ACT boxed IsaacGym: 3 success + 3 failure replays" \
      -f concat -safe 0 -i "$PLAYLIST"
done
