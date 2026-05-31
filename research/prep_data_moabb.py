"""Fetch BCIC-IV-2a (BNCI2014_001) via MOABB and write it in the repo's npz schema.

Produces data/raw/bciciv2a/A{sid:02d}{T,E}.npz with keys matching
data/download.py: eeg (C,T) float64, events (onset sample latencies), labels (1-4),
sfreq. Session 'T' = train session, 'E' = evaluation session.

Usage:
    python research/prep_data_moabb.py                # all 9 subjects
    python research/prep_data_moabb.py --subjects 1   # just subject 1 (validation)
"""

import argparse
from pathlib import Path

import numpy as np
import mne

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "raw" / "bciciv2a"

# MOABB annotation description -> repo class label (1=left,2=right,3=feet,4=tongue)
NAME_TO_LABEL = {
    "left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4,
}


def convert_session(raw: mne.io.BaseRaw, out_path: Path) -> None:
    sfreq = raw.info["sfreq"]
    eeg_channels = [ch for ch in raw.ch_names if not ch.startswith("EOG")][:22]
    raw = raw.copy().pick(eeg_channels)
    data = raw.get_data()  # (22, T)

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    code_to_label = {}
    for name, code in event_id.items():
        for key, lab in NAME_TO_LABEL.items():
            if key in name:
                code_to_label[code] = lab
                break

    latencies, labels = [], []
    for onset, _, code in events:
        if code in code_to_label:
            latencies.append(int(onset))
            labels.append(code_to_label[code])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), eeg=data.astype(np.float64),
             events=np.array(latencies), labels=np.array(labels),
             sfreq=np.float64(sfreq))
    print(f"  saved {out_path.name}  eeg={data.shape} sfreq={sfreq} trials={len(labels)} "
          f"classes={sorted(set(labels))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    args = ap.parse_args()

    from moabb.datasets import BNCI2014_001
    dataset = BNCI2014_001()

    for sid in args.subjects:
        print(f"subject {sid} ...", flush=True)
        data = dataset.get_data(subjects=[sid])[sid]
        # session keys are insertion-ordered: first = train ('T'), second = eval ('E')
        sess_keys = list(data.keys())
        for suffix, skey in zip(["T", "E"], sess_keys):
            runs = data[skey]
            raws = [runs[r] for r in runs]
            raw = mne.concatenate_raws([r.copy() for r in raws], verbose=False)
            convert_session(raw, OUT_DIR / f"A{sid:02d}{suffix}.npz")


if __name__ == "__main__":
    main()
