"""Download helpers for BCIC-IV-2a datasets.

Usage:
    python -m data.download --dataset bciciv2a --data_dir ./data/raw/bciciv2a
"""

import argparse
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# BCIC-IV-2a — .gdf format
# ---------------------------------------------------------------------------

def convert_bciciv2a_gdf(gdf_path: Path, out_path: Path) -> None:
    try:
        import mne
    except ImportError:
        raise ImportError("mne is required: pip install mne")

    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    eeg_channels = [ch for ch in raw.ch_names if not ch.startswith("EOG")][:22]
    raw.pick_channels(eeg_channels)
    data = raw.get_data()

    events, event_id = mne.events_from_annotations(raw, verbose=False)
    class_map = {"769": 1, "770": 2, "771": 3, "772": 4}
    latencies, labels = [], []
    for evt in events:
        for name, code in event_id.items():
            if evt[2] == code:
                for cls_str, cls_int in class_map.items():
                    if cls_str in name:
                        latencies.append(evt[0])
                        labels.append(cls_int)
                        break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), eeg=data.astype(np.float64),
             events=np.array(latencies), labels=np.array(labels),
             sfreq=np.float64(sfreq))
    print(f"  Saved {out_path} ({len(latencies)} events)")


def convert_bciciv2a_directory(data_dir: Path) -> None:
    gdf_files = list(data_dir.glob("*.gdf"))
    if not gdf_files:
        print(f"No .gdf files found in {data_dir}")
        return
    for gdf_path in sorted(gdf_files):
        out_path = gdf_path.with_suffix(".npz")
        print(f"Converting {gdf_path.name} ...")
        try:
            convert_bciciv2a_gdf(gdf_path, out_path)
        except Exception as e:
            print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dataset conversion helper")
    parser.add_argument("--dataset", choices=["bciciv2a"], required=True)
    parser.add_argument("--data_dir", required=True, help="Directory containing .mat/.gdf files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.dataset == "bciciv2a":
        print("BCIC-IV-2a requires manual download:")
        print("  https://www.bbci.de/competition/iv/")
        print(f"Place .gdf files in: {data_dir}")
        print(f"Then re-run: python -m data.download --dataset bciciv2a --data_dir {data_dir}")
        convert_bciciv2a_directory(data_dir)


if __name__ == "__main__":
    main()
