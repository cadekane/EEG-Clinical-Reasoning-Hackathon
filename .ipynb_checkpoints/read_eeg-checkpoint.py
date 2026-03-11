import mne.io as io

file_path = "../koa_scratch/ds004504_eeg/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"

# Read the raw data
raw = io.read_raw_eeglab(file_path, preload=True)

# Print some basic information
print(raw.info)
print(raw.annotations)