"""R029/A4: fixed-TAT full protocol without FeatureAlign; all else matches R001."""
import sys
from run_paper_core import main

if __name__ == "__main__":
    sys.argv[1:1] = [
        "--experiment_id", "R029", "--seed", "42", "--ks", "0",
        "--pt_epochs", "100", "--adapt_epochs", "50", "--batch_size", "16",
        "--lr", "0.001", "--adapt_lr", "0.0005",
        "--tat_weight", "0.03", "--topology_mode", "correct",
        "--adapt_mode", "full", "--use_feature_align", "0",
        "--spec_mode", "D", "--attn_gate", "none",
        "--out_dir", "results/R029_no_featurealign_s42",
    ]
    main()
