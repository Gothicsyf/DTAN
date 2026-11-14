"""R018/T0: weight-zero TAT control; keeps the second forward for BN fairness."""
import sys
from run_paper_core import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--experiment_id", "R018", "--seed", "42", "--ks", "0", "--tat_weight", "0", "--topology_mode", "correct", "--adapt_mode", "full", "--use_feature_align", "1", "--spec_mode", "D", "--attn_gate", "none", "--out_dir", "results/R018_no_tat_s42"]
    main()
