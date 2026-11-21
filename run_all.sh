#!/bin/bash
# Reproduce all simulation results (30 replications each). Several hours on two cores.
cd "$(dirname "$0")"
for e in layout S0 S1 S1w S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S12 S13 V1; do
  python3 experiments.py $e --reps 30
done
python3 s12b.py
python3 s_v3.py
python3 s_v9.py
python3 paired_stats.py
python3 paper_figs.py
