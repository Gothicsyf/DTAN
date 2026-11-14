"""R028/A3: naive concatenation replaces DBPF; all training settings match R001."""
import sys
import run_paper_core as core
from Models.D_STNet_SimpleConcat import Net

if __name__ == "__main__":
    core.base.Net = Net
    sys.argv[1:1] = ["--experiment_id", "R028", "--seed", "42", "--ks", "0",
        "--pt_epochs", "100", "--adapt_epochs", "50", "--batch_size", "16",
        "--lr", "0.001", "--adapt_lr", "0.0005", "--tat_weight", "0.03",
        "--topology_mode", "correct", "--adapt_mode", "full", "--use_feature_align", "1",
        "--spec_mode", "D", "--attn_gate", "none", "--out_dir", "results/R028_simple_concat_s42"]
    core.main()
