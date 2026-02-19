#!/bin/bash
cd ~/workspace/polymarket-predictor
cp results.json docs/results.json 2>/dev/null
git add -A
git commit -m "update results $(date -u +%H:%M)" 2>/dev/null
git push origin main 2>/dev/null
