# HGD environment and reproducible validation

Host: SSH alias myserver. GPU 0: NVIDIA L40, approximately 46 GB.
Reuse existing /home/u24600002150531/.conda/envs/DST/bin/python.
Observed versions: torch 2.5.1+cu124; numpy 1.26.4; scipy 1.17.1; mne 1.12.1.
No dependencies installed or upgraded. screen is absent, tmux is present.
No prior .aris/compute ledger was found in the project or its parent.

Validation command (run verbatim):

ssh -o BatchMode=yes -o ConnectTimeout=15 myserver 'cd /home/u24600002150531/8410/xhl/MyCode-main/D-syf-HGD-20260921 && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 /home/u24600002150531/.conda/envs/DST/bin/python smoke_hgd.py --synthetic'

Expected: HGD_WITNESS_PASS; all three models yield [16,4], finite backward gradients; identical four-class TAT numeric matrix, dataset subject count 14. Writes validation/kernel_witness.json, no training checkpoint.

This validates the existing environment and dimension port, not data integrity or full training; those have separate gates.
