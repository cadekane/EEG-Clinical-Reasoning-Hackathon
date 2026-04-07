"""
01_preprocess_45hz.py
─────────────────────
Apply bandpass filter (0.5–45 Hz) to raw EEG .set files and save to
ds004504_45hz/ folder inside the dataset directory.

Input:  {BASE_DIR}/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set
Output: {BASE_DIR}/ds004504_45hz/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg_45hz.set

Requires: eeglabio  (pip install eeglabio)
"""

import mne
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
mne.set_log_level("WARNING")

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR         = "/mnt/lustre/koa/scratch/rikimacm/ds004504_eeg"
OUTPUT_DIR       = os.path.join(BASE_DIR, "ds004504_45hz")
PARTICIPANTS_FILE = os.path.join(BASE_DIR, "participants.tsv")

LOWCUT        = 0.5    # Hz
HIGHCUT       = 45.0   # Hz
FILTER_ORDER  = 4
# ──────────────────────────────────────────────────────────────────────────────


def preprocess_subject(sub_id: str) -> bool:
    in_path = os.path.join(BASE_DIR, sub_id, "eeg",
                           f"{sub_id}_task-eyesclosed_eeg.set")

    if not os.path.exists(in_path):
        print(f"[SKIP] {sub_id}: raw file not found at {in_path}")
        return False

    out_dir  = os.path.join(OUTPUT_DIR, sub_id, "eeg")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sub_id}_task-eyesclosed_eeg_45hz.set")

    if os.path.exists(out_path):
        print(f"[SKIP] {sub_id}: already processed")
        return True

    try:
        print(f"[PROC] {sub_id} ...")
        raw = mne.io.read_raw_eeglab(in_path, preload=True)

        # Bandpass filter 0.5–45 Hz (Butterworth, order 4)
        raw.filter(
            l_freq=LOWCUT,
            h_freq=HIGHCUT,
            method="iir",
            iir_params=dict(order=FILTER_ORDER, ftype="butter"),
            picks="eeg",
        )

        # Re-reference to common average
        raw.set_eeg_reference("average", projection=False)

        # Export as EEGLAB .set (requires eeglabio: pip install eeglabio)
        raw.export(out_path, fmt="eeglab", overwrite=True)
        duration_min = raw.times[-1] / 60
        print(f"[DONE] {sub_id}  ({duration_min:.1f} min)  -> {out_path}")
        return True

    except Exception as exc:
        print(f"[ERROR] {sub_id}: {exc}")
        return False


def main():
    participants = pd.read_csv(PARTICIPANTS_FILE, sep="\t")
    success, failed = 0, []

    for _, row in participants.iterrows():
        sub_id = row["participant_id"]
        if preprocess_subject(sub_id):
            success += 1
        else:
            failed.append(sub_id)

    print(f"\n=== Preprocessing complete: {success}/{len(participants)} ===")
    if failed:
        print(f"Failed subjects: {failed}")


if __name__ == "__main__":
    main()
