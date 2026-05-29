"""Download helpers for Jeong2020 and BCIC-IV-2a datasets.

Usage:
    # After downloading mrk-and-cnt_datasets.tar.gz from GigaDB:
    python -m data.download --dataset jeong2020 --data_dir ./data/raw/jeong2020 --tar ./mrk-and-cnt_datasets.tar.gz

    # Or if already extracted:
    python -m data.download --dataset jeong2020 --data_dir ./data/raw/jeong2020 --convert-only

    python -m data.download --dataset bciciv2a  --data_dir ./data/raw/bciciv2a --convert-only
"""

import argparse
import tarfile
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Jeong 2020 — Neuroscan .cnt + .mrk format
# ---------------------------------------------------------------------------

# Jeong 2020 marker codes → 0-indexed class label
# The dataset has 11 movement tasks: ME and MI for 5 movements + rest
# Marker codes from the paper/dataset documentation:
#   1=ME elbow flexion, 2=ME elbow extension, 3=ME forearm sup, 4=ME forearm pron,
#   5=ME wrist flex, 6=ME wrist ext, 7=ME hand open, 8=ME hand close,
#   9=ME rest, 10=MI (same movements), 11=rest
# The exact codes vary; we auto-detect from the .mrk file and sort them.
JEONG_N_CLASSES = 11


def parse_mrk_file(mrk_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Neuroscan .mrk marker file.

    .mrk files are tab-separated text with columns:
        offset  type  label  (and possibly more)

    Returns:
        latencies: (N,) sample indices (0-indexed)
        labels:    (N,) integer class labels (0-indexed)
    """
    latencies, raw_types = [], []

    with open(mrk_path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                latencies.append(int(parts[0]))
                raw_types.append(parts[1])
            except ValueError:
                continue

    if not latencies:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Map marker types to 0-indexed labels, sorted for reproducibility
    unique_types = sorted(set(raw_types), key=lambda x: (len(x), x))
    type_to_label = {t: i for i, t in enumerate(unique_types)}
    labels = np.array([type_to_label[t] for t in raw_types], dtype=np.int32)

    return np.array(latencies, dtype=np.int64), labels


def convert_jeong2020_paired_mats(cnt_path: Path, mrk_path: Path, out_path: Path) -> None:
    """Convert paired cnt_*.mat + mrk_*.mat files to .npz.

    Jeong2020 structure:
        S{subject}/Day{session}/
            cnt_{condition}({run}).mat  — EEG data
            mrk_{condition}({run}).mat  — event markers
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")

    # Read EEG from cnt file

    def find_largest_dataset(group, prefix=""):
        """Recursively find largest dataset in HDF5 group, returning (path_list, dataset_ref)."""
        largest_path = None
        largest_obj = None
        largest_size = 0
        for key in group.keys():
            if key.startswith('#'):
                continue
            obj = group[key]
            if isinstance(obj, h5py.Dataset) and obj.size > largest_size:
                largest_size = obj.size
                largest_path = [key]
                largest_obj = obj
            elif isinstance(obj, h5py.Group):
                sub_path, sub_obj = find_largest_dataset(obj, prefix=f"{prefix}/{key}" if prefix else key)
                if sub_obj is not None and sub_obj.size > largest_size:
                    largest_size = sub_obj.size
                    largest_path = [key] + sub_path
                    largest_obj = sub_obj
        return largest_path or [], largest_obj

    def get_nested_dataset(group, path_list):
        """Traverse nested path in HDF5 group."""
        obj = group
        for key in path_list:
            obj = obj[key]
        return obj[()]

    with h5py.File(str(cnt_path), 'r') as f:
        eeg_data = None

        # Try common key names
        for key_name in ['cnt', 'data', 'EEG']:
            if key_name in f:
                obj = f[key_name]
                if isinstance(obj, h5py.Dataset):
                    eeg_data = obj[()]
                    break
                elif isinstance(obj, h5py.Group):
                    dataset_path, _ = find_largest_dataset(obj)
                    if dataset_path:
                        eeg_data = get_nested_dataset(obj, dataset_path)
                        break

        # If still not found, find largest anywhere in file
        if eeg_data is None:
            dataset_path, _ = find_largest_dataset(f)
            if dataset_path:
                eeg_data = get_nested_dataset(f, dataset_path)

        if eeg_data is None:
            raise ValueError(f"Could not find EEG data in {cnt_path}. Keys: {list(f.keys())}")

        # Transpose if needed (Fortran order: more rows than cols means (T, C))
        if eeg_data.ndim == 2 and eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T

        # Default sfreq for Jeong2020
        sfreq = 2500.0

    # Read markers from mrk file
    with h5py.File(str(mrk_path), 'r') as f:
        latencies = np.array([], dtype=int)
        labels = np.array([], dtype=int)

        # Try to extract from 'mrk' group or dataset
        if 'mrk' in f:
            mrk_obj = f['mrk']
            if isinstance(mrk_obj, h5py.Dataset):
                # Direct dataset → treat as labels
                try:
                    labels = np.array([int(e) - 1 for e in mrk_obj[()].flatten()], dtype=int)
                except (ValueError, TypeError):
                    unique_types = sorted(set(str(e) for e in mrk_obj[()].flatten()))
                    type_map = {t: i for i, t in enumerate(unique_types)}
                    labels = np.array([type_map[str(e)] for e in mrk_obj[()].flatten()], dtype=int)
            elif isinstance(mrk_obj, h5py.Group):
                # Group with latency/event subkeys
                if 'time' in mrk_obj:
                    latencies = np.array(mrk_obj['time'][()], dtype=int).flatten() - 1
                elif 'latency' in mrk_obj:
                    latencies = np.array(mrk_obj['latency'][()], dtype=int).flatten() - 1

                # Labels from 'y' (one-hot encoded) or 'event' (if Dataset)
                if 'y' in mrk_obj:
                    y_data = mrk_obj['y'][()]
                    if y_data.ndim == 2:
                        labels = np.argmax(y_data, axis=1)
                    else:
                        labels = y_data.flatten()
                elif 'event' in mrk_obj:
                    event_obj = mrk_obj['event']
                    if isinstance(event_obj, h5py.Dataset):
                        event_data = event_obj[()]
                        try:
                            labels = np.array([int(e) - 1 for e in event_data.flatten()], dtype=int)
                        except (ValueError, TypeError):
                            unique_types = sorted(set(str(e) for e in event_data.flatten()))
                            type_map = {t: i for i, t in enumerate(unique_types)}
                            labels = np.array([type_map[str(e)] for e in event_data.flatten()], dtype=int)

    # If no explicit events, treat entire trial as one event
    if len(latencies) == 0:
        latencies = np.array([0], dtype=int)
    if len(labels) == 0:
        labels = np.array([0], dtype=int)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), eeg=eeg_data.astype(np.float64),
             events=latencies, labels=labels, sfreq=np.float64(sfreq))
    print(f"    Saved {out_path.name}  ({eeg_data.shape[0]}ch, {eeg_data.shape[1]}samples)")


def extract_tar(tar_path: Path, out_dir: Path) -> None:
    """Extract tar.gz into out_dir."""
    print(f"Extracting {tar_path.name} → {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "r:gz") as tf:
        tf.extractall(str(out_dir))
    print("Extraction complete.")


def convert_jeong2020_directory(data_dir: Path) -> None:
    """Find all cnt_*.mat files and pair with mrk_*.mat, convert to .npz.

    Expected structure:
        data_dir/
            S{subject}/
                Day{session}/
                    cnt_{condition}({run}).mat
                    mrk_{condition}({run}).mat
    """
    cnt_files = sorted(data_dir.rglob("cnt_*.mat"))
    if not cnt_files:
        print(f"No cnt_*.mat files found under {data_dir}")
        return

    print(f"Found {len(cnt_files)} cnt_*.mat files.")
    errors = []
    for cnt_path in cnt_files:
        # Find paired mrk file (same base name)
        mrk_path = cnt_path.parent / cnt_path.name.replace("cnt_", "mrk_")
        if not mrk_path.exists():
            print(f"  WARNING: No matching {mrk_path.name} for {cnt_path.name}")
            continue

        out_path = cnt_path.with_stem(cnt_path.stem.replace("cnt_", "")).with_suffix(".npz")
        if out_path.exists():
            print(f"  Skipping {out_path.name} (already converted)")
            continue

        print(f"Converting {cnt_path.name} + {mrk_path.name} ...")
        try:
            convert_jeong2020_paired_mats(cnt_path, mrk_path, out_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((cnt_path, e))

    if errors:
        print(f"\n{len(errors)} file(s) failed:")
        for p, e in errors:
            print(f"  {p.name}: {e}")
    else:
        print("\nAll files converted successfully.")

    # Print first .npz to verify structure
    npz_files = sorted(data_dir.rglob("*.npz"))
    if npz_files:
        d = np.load(str(npz_files[0]), allow_pickle=True)
        print(f"\nSample check ({npz_files[0].name}):")
        print(f"  eeg shape : {d['eeg'].shape}")
        print(f"  events    : {d['events'].shape}  first few: {d['events'][:5]}")
        print(f"  labels    : {d['labels'].shape}  unique: {np.unique(d['labels'])}")
        print(f"  sfreq     : {d['sfreq']}")


def infer_jeong_sessions(data_dir: Path) -> None:
    """Print a summary of what sessions/subjects were found after extraction."""
    npz_files = sorted(data_dir.rglob("*.npz"))
    if not npz_files:
        return
    print(f"\nConverted files ({len(npz_files)} total):")
    for f in npz_files[:10]:
        print(f"  {f.relative_to(data_dir)}")
    if len(npz_files) > 10:
        print(f"  ... and {len(npz_files) - 10} more")


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
    parser.add_argument("--dataset", choices=["jeong2020", "bciciv2a"], required=True)
    parser.add_argument("--data_dir", required=True, help="Directory containing .mat/.gdf files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.dataset == "jeong2020":
        print(f"Converting Jeong2020 .mat files in {data_dir} ...")
        convert_jeong2020_directory(data_dir)
        infer_jeong_sessions(data_dir)

    elif args.dataset == "bciciv2a":
        print("BCIC-IV-2a requires manual download:")
        print("  https://www.bbci.de/competition/iv/")
        print(f"Place .gdf files in: {data_dir}")
        print("Then re-run: python -m data.download --dataset bciciv2a --data_dir {data_dir}")
        convert_bciciv2a_directory(data_dir)


if __name__ == "__main__":
    main()
