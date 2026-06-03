"""Preprocess and cache EEG datasets on the Modal volume.

Run once per dataset before training — saves preprocessed (X, y) arrays so
experiment jobs skip the expensive resample/filter/epoch pipeline entirely.

Usage:
    modal run scripts/modal_prep_cache.py --dataset bciciv2a
"""

import modal

app = modal.App("eeg-prep-cache")

data_volume = modal.Volume.from_name("eeg-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.10",
        "mne>=1.6",
        "h5py>=3.9",
    )
    .add_local_python_source("data", copy=True)
)

CACHE_DIRS = {
    "bciciv2a":        "/data/bciciv2a_cache",
    "bciciv2a_labram": "/data/bciciv2a_labram_cache",  # wider bandpass, no z-score (µV for LaBraM)
}

DATA_DIRS = {
    "bciciv2a":        "/data/bciciv2a",
    "bciciv2a_labram": "/data/bciciv2a",  # same raw data, different preprocessing
}


@app.function(
    image=image,
    volumes={"/data": data_volume},
    timeout=7200,
    cpu=8,
)
def build_cache(dataset_name: str) -> None:
    from data.datasets import BCICIVDataset
    from data.preprocessing import LABRAM_PREPROCESS_CONFIG

    if dataset_name == "bciciv2a":
        dataset = BCICIVDataset(DATA_DIRS["bciciv2a"])
    elif dataset_name == "bciciv2a_labram":
        dataset = BCICIVDataset(
            DATA_DIRS["bciciv2a_labram"],
            preprocess_config=LABRAM_PREPROCESS_CONFIG,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cache_dir = CACHE_DIRS[dataset_name]
    print(f"Building cache for {dataset_name} → {cache_dir}")
    dataset.save_preprocessed_cache(cache_dir)
    data_volume.commit()
    print("Done.")


@app.local_entrypoint()
def main(dataset: str = "bciciv2a") -> None:
    call = build_cache.spawn(dataset)
    print(f"Spawned cache build: {call.object_id}")
    print("Safe to disconnect — job will continue on Modal.")
