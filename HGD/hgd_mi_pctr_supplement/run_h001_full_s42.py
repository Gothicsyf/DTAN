import sys
from run_paper_core import main
if __name__ == "__main__":
    sys.argv[1:1] = ['--seed', '42', '--ks', '0', '--pt_epochs', '100', '--adapt_epochs', '50', '--batch_size', '16', '--lr', '0.001', '--adapt_lr', '0.0005', '--tat_weight', '0.03', '--topology_mode', 'correct', '--adapt_mode', 'full', '--lambda_ent', '0.3', '--use_feature_align', '1', '--spec_mode', 'D', '--attn_gate', 'none', '--data_processed', '/home/u24600002150531/8410/data/BCI/HGD/processed/HGD_ea', '--out_dir', 'results/H001_full_s42']
    main()
