"""Plot experiment results — one figure per dataset/backbone combo.

Auto-discovers all modal_summary.json files under results/.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

ADAPTATION = {'finetune', 'lora', 'ea_lora'}

COLORS = {
    'loso':     '#1f77b4',
    'ea':       '#ff7f0e',
    'tta':      '#2ca02c',
    'finetune': '#d62728',
    'lora':     '#9467bd',
    'ea_lora':  '#8c564b',
    'cld':      '#e377c2',
    'ea_cld':   '#17becf',
}
MARKERS = {'finetune': 'o', 'lora': 's', 'ea_lora': '^', 'cld': 'D', 'ea_cld': 'P'}
LABELS = {
    'loso': 'LOSO', 'ea': 'EA', 'tta': 'TTA',
    'finetune': 'Fine-tune', 'lora': 'LoRA', 'ea_lora': 'EA+LoRA',
    'cld': 'CLD', 'ea_cld': 'EA+CLD',
}
DATASET_LABELS = {
    'bciciv2a': 'BCI Competition IV 2a',
    'jeong2020': 'Jeong 2020',
    'synthetic': 'Synthetic',
}
BACKBONE_LABELS = {
    'eegnet': 'EEGNet',
    'shallowconv': 'ShallowConvNet',
    'conformer': 'EEG-Conformer',
}


def load_results(json_path: str) -> pd.DataFrame:
    with open(json_path) as f:
        raw = json.load(f)
    rows = []
    for _, k_entries in raw.items():
        for k_str, entry in k_entries.items():
            rows.append({
                'method':     entry['_meta']['method'],
                'subject_id': entry['subject_id'],
                'k_minutes':  float(k_str),
                'bca':        entry['bca'],
            })
    df = pd.DataFrame(rows)
    return (
        df.groupby(['method', 'k_minutes'])['bca']
        .agg(mean_bca='mean', std_bca='std')
        .reset_index()
    )


for json_path in sorted(Path('results').glob('*/*/modal_summary.json')):
    backbone = json_path.parent.name
    dataset = json_path.parent.parent.name

    df = load_results(json_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Source-only methods: horizontal dashed lines
    for method in ['loso', 'ea', 'tta']:
        d = df[(df['method'] == method) & (df['k_minutes'] == 0.0)]
        if len(d) == 0:
            continue
        mean = d['mean_bca'].values[0]
        std = d['std_bca'].values[0]
        ax.axhline(mean, color=COLORS[method], linestyle='--', linewidth=2,
                   label=LABELS[method], alpha=0.85)
        if not np.isnan(std):
            ax.axhspan(mean - std, mean + std, color=COLORS[method], alpha=0.08)

    # Adaptation methods: learning curves
    for method in ['finetune', 'lora', 'ea_lora', 'cld', 'ea_cld']:
        d = df[df['method'] == method].sort_values('k_minutes')
        if len(d) == 0:
            continue
        k_vals = d['k_minutes'].values
        means = d['mean_bca'].values
        stds = d['std_bca'].values
        ax.plot(k_vals, means, marker=MARKERS[method], markersize=8,
                linewidth=2.5, label=LABELS[method], color=COLORS[method], alpha=0.9)
        ax.fill_between(k_vals, means - stds, means + stds,
                        alpha=0.15, color=COLORS[method])

    ax.axhline(0.25, color='gray', linestyle=':', linewidth=1.2, label='Chance (25%)')
    ax.set_xlabel('Minutes of Target Calibration Data', fontsize=13, fontweight='bold')
    ax.set_ylabel('Balanced Class Accuracy (BCA)', fontsize=13, fontweight='bold')
    ds_label = DATASET_LABELS.get(dataset, dataset)
    bb_label = BACKBONE_LABELS.get(backbone, backbone)
    ax.set_title(f'{bb_label} — {ds_label}\nAdaptation Methods vs Calibration Time',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0.2, 1.0])
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9,
              ncol=2, title='Method', title_fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out_path = json_path.parent / 'comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved: {out_path}')
    plt.close()
