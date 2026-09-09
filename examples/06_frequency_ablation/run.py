"""从各频率的归档合同恢复参数，避免命令行默认值覆盖原实验设定。"""
from dataclasses import fields
import json
from pathlib import Path
from multislice_xyz_tt_study.multislice_tt import MultiSliceConfig, run_experiment


def main():
    example = Path(__file__).resolve().parent
    known = {item.name for item in fields(MultiSliceConfig)}
    for frequency in (100, 150, 300):
        report = json.loads((example/f'results/metrics/aux_frequency/report_{frequency}khz.json').read_text())
        values = {k: v for k, v in report['contract'].items() if k in known}
        for key in ('sensor_z_m', 'center_shape', 'evaluation_shape', 'norm_shape'):
            values[key] = tuple(values[key])
        for key in ('snapshot', 'baseline'):
            values[key] = str(example/'data'/Path(values[key]).name)
        values['output'] = str(example/f'outputs/{frequency}khz')
        run_experiment(MultiSliceConfig(**values))


if __name__ == '__main__':
    main()
