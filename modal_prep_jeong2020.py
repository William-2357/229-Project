"""Extract + convert Jeong2020 on Modal.

Workflow (two steps):

  # 1. Upload the 10 GB tar.gz to the Modal volume (one-time)
  modal volume put eeg-data mrk-and-cnt_datasets.tar.gz /jeong2020_tar.tar.gz

  # 2. Run extraction + conversion on Modal (fast — datacenter CPU/disk)
  modal run modal_prep_jeong2020.py

After step 2, /jeong2020/ inside the volume contains all the .npz files
ready for use by the experiment runner.
"""

import modal

app = modal.App("eeg-prep-jeong2020")

data_volume = modal.Volume.from_name("eeg-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.24", "h5py>=3.9", "scipy>=1.10")
    .add_local_python_source("data", copy=True)
)


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=7200,
    cpu=8,
)
def extract_and_convert():
    import tarfile
    import numpy as np
    from pathlib import Path
    from data.download import convert_jeong2020_directory

    tar_path = Path("/data/jeong2020_tar.tar.gz")
    out_dir = Path("/data/jeong2020")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tar_path.exists():
        raise FileNotFoundError(
            f"Expected {tar_path}. Upload first with:\n"
            "  modal volume put eeg-data mrk-and-cnt_datasets.tar.gz /jeong2020_tar.tar.gz"
        )

    if any(out_dir.glob("S*/Day*/*.mat")) or any(out_dir.glob("S*/Day*/*.npz")):
        print("Extracted .mat/.npz files already present — skipping tar extraction.")
    else:
        print(f"Extracting {tar_path} ({tar_path.stat().st_size / 1e9:.1f} GB) ...")
        with tarfile.open(str(tar_path), "r:gz") as tf:
            tf.extractall(str(out_dir))
        print("Extraction done.")

    print("Converting .mat → .npz (skips already-converted files) ...")
    convert_jeong2020_directory(out_dir)
    print("Conversion done.")

    # Restructure: S{N}/Day{1,2}/*.npz  →  s{NN}/s{NN}_session{1,2}.npz
    # Concatenate per-run npz files into a single session npz
    print("\nRestructuring into JeongDataset layout ...")
    n_done = 0
    for subj_dir in sorted(out_dir.glob("S*")):
        if not subj_dir.is_dir():
            continue
        # Subject number from "S1", "S10", etc.
        try:
            subj_num = int(subj_dir.name[1:])
        except ValueError:
            continue

        new_subj_dir = out_dir / f"s{subj_num:02d}"
        new_subj_dir.mkdir(exist_ok=True)

        for day_dir in sorted(subj_dir.glob("Day*")):
            try:
                day_num = int(day_dir.name[3:])
            except ValueError:
                continue

            run_npzs = sorted(day_dir.glob("*.npz"))
            if not run_npzs:
                continue

            eegs, events_list, labels_list, sfreq = [], [], [], None
            offset = 0
            for run_path in run_npzs:
                d = np.load(run_path, allow_pickle=True)
                eeg = d["eeg"]                   # (C, T)
                events = d["events"].astype(int)
                labels = d["labels"].astype(int)
                sfreq = float(d["sfreq"])

                eegs.append(eeg)
                events_list.append(events + offset)
                labels_list.append(labels)
                offset += eeg.shape[1]

            merged_eeg = np.concatenate(eegs, axis=1)
            merged_events = np.concatenate(events_list)
            merged_labels = np.concatenate(labels_list)

            out_path = new_subj_dir / f"s{subj_num:02d}_session{day_num}.npz"
            np.savez(out_path, eeg=merged_eeg, events=merged_events,
                     labels=merged_labels, sfreq=np.float64(sfreq))
            n_done += 1
            print(f"  → {out_path.name} "
                  f"({merged_eeg.shape[0]}ch × {merged_eeg.shape[1]} samples, "
                  f"{len(merged_labels)} events)")

    print(f"\nRestructure done: {n_done} session files written.")

    # Commit volume so changes persist
    data_volume.commit()
    print(f"\nDone. Volume contents at /jeong2020/")


@app.local_entrypoint()
def main():
    extract_and_convert.remote()
