"""重算 19 个窗口；显式区分归档的跨求解器基线与新的匹配基线。"""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['reference', 'matched'], default='reference')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    example = Path(__file__).resolve().parent
    baseline = 'provisional_pytorch_baseline.h5' if args.mode == 'reference' else 'incflo_baseline_t30p969.h5'
    status = 'provisional_cross_solver' if args.mode == 'reference' else 'matched_incflo'
    command = [sys.executable, '-m', 'incflo_multiring_tt_study.run_multiwindow',
               '--snapshot', str(example/'data/incflo_snapshot_t30p969.h5'),
               '--baseline', str(example/'data'/baseline),
               '--output', str(example/'outputs'/args.mode), '--baseline-status', status,
               '--basis', 'dense', '--sensors-per-ring', '48', '--noise-repeats', '4',
               '--center-z', *[f'{1.35+0.3*i:.2f}' for i in range(19)]]
    subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
