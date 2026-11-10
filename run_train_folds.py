"""Run only the complete DPMG model, sequentially, with explicit settings."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


def parse_folds(value):
    folds = []
    try:
        for part in value.split(','):
            if '-' in part:
                first, last = map(int, part.split('-'))
                if first > last:
                    raise ValueError
                folds.extend(range(first, last + 1))
            else:
                folds.append(int(part))
        if not folds or any(f < 0 or f > 31 for f in folds):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('Use folds 0-31, e.g. 0, 0-31, or 0,2-5.')
    return list(dict.fromkeys(folds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds', type=parse_folds, default=parse_folds('0-31'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--output_dir', default='results/dpmg_main')
    parser.add_argument('--seed', type=int, default=12000)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--save_model', choices=['T', 'F'], default='T')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_dir).resolve()
    for fold in args.folds:
        command = [sys.executable, str(root / 'HIST2ST_pretrain_finetune.py'),
                   '--data', 'her2st', '--gpu', str(args.gpu), '--fold', str(fold),
                   '--seed', str(args.seed), '--data_root', str(data_root),
                   '--pretrain_epochs', str(args.pretrain_epochs),
                   '--finetune_epochs', str(args.finetune_epochs),
                   '--gene_list_file', str(root / 'recommended_gene_list_her2st.txt'),
                   '--disease_prior_file', str(root / 'disease_gene_prior_her2st1.csv'),
                   '--disease_relation_file', str(root / 'disease_gene_relation_her2st1.csv'),
                   '--lr', '1e-5', '--dropout', '0.2', '--zinb', '0', '--nb', 'F',
                   '--bake', '0', '--corr_lambda', '0', '--use_feature_gate', 'T',
                   '--attn_reduction', '4', '--use_disease_prior', 'T',
                   '--prior_corr_lambda', '0.001', '--neighbor', '4',
                   '--prune', 'Grid', '--policy', 'mean', '--tag', '5-7-2-8-4-16-32',
                   '--save_model', args.save_model, '--output_dir', str(output_root)]
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        fold_dir = output_root / f'fold_{fold:02d}'
        # Never overwrite a previous run or infer success from partial CSV output.
        if fold_dir.exists() and any(fold_dir.iterdir()):
            raise SystemExit(f'Output already exists: {fold_dir}. Use a new --output_dir or select remaining --folds.')
        subprocess.run(command, cwd=root, check=True)
        (fold_dir / 'completed.json').write_text(
            json.dumps({'command': command, 'fold': fold}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
