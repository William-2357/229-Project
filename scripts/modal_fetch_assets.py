"""Fetch raw data + pretrained checkpoints straight onto the eeg-data Modal volume.

Runs on Modal (controlled deps, fast network, writes directly to the volume — no
multi-GB uploads from a laptop).

    modal run scripts/modal_fetch_assets.py --what checkpoints   # 3 public HF checkpoints
    modal run scripts/modal_fetch_assets.py --what data          # BCICIV2a via MOABB -> A0XT.npz
    modal run scripts/modal_fetch_assets.py --what all

LaBraM (935963004/LaBraM) is gated/private and is NOT fetched here — request access
on huggingface.co with your account, then upload it manually:
    modal volume put eeg-data labram-base.pth /labram-base.pth
"""

import modal

app = modal.App("eeg-fetch-assets")
data_volume = modal.Volume.from_name("eeg-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24,<2.0",
        "scipy>=1.10",
        "mne>=1.6",
        "moabb>=1.1",
        "huggingface_hub>=0.23",
        "requests>=2.31",
    )
)

# HF repo -> (filename in repo, destination path on /data)
CHECKPOINTS = {
    "starself/MIRepNet":   ("MIRepNet.pth",                       "/data/MIRepNet.pth"),
    "wenhuic/Neuro-GPT":   ("pretrained_model/pytorch_model.bin", "/data/neuro_gpt.pt"),
    "weighting666/CBraMod": ("pretrained_weights.pth",            "/data/CBraMod_checkpoint.pth"),
}


@app.function(image=image, volumes={"/data": data_volume}, timeout=3600)
def fetch_checkpoints() -> None:
    import os
    import shutil
    from huggingface_hub import hf_hub_download

    for repo, (fname, dest) in CHECKPOINTS.items():
        if os.path.exists(dest):
            print(f"[skip] {dest} already exists ({os.path.getsize(dest)/1e6:.1f} MB)")
            continue
        print(f"[get ] {repo}:{fname} -> {dest}", flush=True)
        src = hf_hub_download(repo_id=repo, filename=fname)
        shutil.copy(src, dest)
        print(f"[done] {dest} ({os.path.getsize(dest)/1e6:.1f} MB)", flush=True)

    data_volume.commit()
    print("Checkpoints committed to eeg-data.")


# Cue-annotation description -> integer label expected by data/datasets.py
# (BCICIVDataset.CLASS_REMAP then maps 1..4 -> 0..3).
_LABEL_MAP = {
    "left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4,
    "769": 1, "770": 2, "771": 3, "772": 4,
}


@app.function(image=image, volumes={"/data": data_volume}, timeout=7200, cpu=4)
def fetch_data() -> None:
    """Download BCICIV2a (MOABB BNCI2014_001) and write A0XT.npz / A0XE.npz.

    Output format matches data/download.py / BCICIVDataset._load_raw_subject:
        eeg    : (22, T) float64 at 250 Hz, EEG channels only
        events : (N,) int  cue-onset sample indices into the concatenated session
        labels : (N,) int  in {1,2,3,4}
        sfreq  : float64
    Runs within a session are concatenated; event latencies are offset by the
    cumulative sample count so they index the concatenated array.
    """
    import os
    import numpy as np
    import mne
    from moabb.datasets import BNCI2014_001

    out_dir = "/data/bciciv2a"
    os.makedirs(out_dir, exist_ok=True)
    ds = BNCI2014_001()
    # MOABB places event onsets at TRIAL START; the motor-imagery window is
    # ds.interval = [2, 6] s. The original npz format (data/download.py) instead
    # marks CUE ONSET (t=2 s), and BCICIVDataset epochs [event, event+4 s] = the
    # [2, 6] s imagery period. So shift every onset forward by interval[0] seconds
    # to land on the cue — without this, epochs capture fixation and loso collapses
    # to chance (validated: 0.26 vs 0.74 reference).
    cue_offset_sec = float(ds.interval[0])

    for subject in range(1, 10):
        data = ds.get_data(subjects=[subject])[subject]
        # sessions: typically {"0train": {...runs}, "1test": {...runs}}
        for sess_name in sorted(data.keys()):
            suffix = "T" if ("train" in sess_name.lower() or sess_name.endswith("T")
                             or sess_name in ("0train", "session_T")) else "E"
            runs = data[sess_name]

            eeg_parts, ev_lat, ev_lab = [], [], []
            offset = 0
            sfreq = None
            for run_name in sorted(runs.keys()):
                raw = runs[run_name]
                sfreq = raw.info["sfreq"]
                eeg_picks = [ch for ch in raw.ch_names if not ch.upper().startswith("EOG")][:22]
                raw_eeg = raw.copy().pick(eeg_picks)
                arr = raw_eeg.get_data()  # (22, T_run)

                events, event_id = mne.events_from_annotations(raw, verbose=False)
                id_to_desc = {v: k for k, v in event_id.items()}
                cue_shift = int(round(cue_offset_sec * sfreq))
                for onset_samp, _, code in events:
                    desc = str(id_to_desc[code]).strip().lower()
                    if desc in _LABEL_MAP:
                        ev_lat.append(int(onset_samp) + cue_shift + offset)
                        ev_lab.append(_LABEL_MAP[desc])

                eeg_parts.append(arr)
                offset += arr.shape[1]

            if not ev_lat:
                print(f"[warn] A{subject:02d}{suffix}: no labeled cues found "
                      f"(annotations: {sorted(set())}) — skipping", flush=True)
                continue

            eeg = np.concatenate(eeg_parts, axis=1).astype(np.float64)
            out_path = os.path.join(out_dir, f"A{subject:02d}{suffix}.npz")
            np.savez(
                out_path,
                eeg=eeg,
                events=np.asarray(ev_lat, dtype=int),
                labels=np.asarray(ev_lab, dtype=int),
                sfreq=np.float64(sfreq),
            )
            print(f"[done] {out_path}  eeg={eeg.shape} n_trials={len(ev_lat)} "
                  f"labels={sorted(set(ev_lab))} sfreq={sfreq}", flush=True)

    data_volume.commit()
    print("Raw data committed to eeg-data:/bciciv2a")


@app.local_entrypoint()
def main(what: str = "all") -> None:
    if what in ("checkpoints", "all"):
        fetch_checkpoints.remote()
    if what in ("data", "all"):
        fetch_data.remote()
