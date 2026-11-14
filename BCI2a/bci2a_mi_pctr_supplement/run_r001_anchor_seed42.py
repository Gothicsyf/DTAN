"""R001: exact 74.31 configuration, seed 42, k=0."""
import sys
from run_paper_core import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--experiment_id", "R001", "--seed", "42", "--ks", "0", "--tat_weight", "0.03", "--topology_mode", "correct", "--adapt_mode", "full", "--use_feature_align", "1", "--spec_mode", "D", "--attn_gate", "none", "--out_dir", "results/R001_anchor_seed42"]
    main()
