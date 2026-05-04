"""
EEGLAB .set v7.3 loader

mne.io.read_raw_eeglab() uses scipy.io.loadmat under the hood for some
operations, which cannot read MATLAB v7.3 (HDF5-based) files. ds005048's
.set files are saved as v7.3, so MNE raises:

    NotImplementedError: Please use HDF reader for matlab v7.3 files, e.g. h5py

This helper bypasses MNE's loader entirely and constructs an mne.io.RawArray
from the EEG struct using pymatreader, which transparently handles both
v7 and v7.3 formats.

Two storage layouts are supported:

  1. Data embedded in the .set file:
        EEG.data is a 2D float array (n_channels, n_times)

  2. Data in a separate .fdt sibling file (classic EEGLAB layout):
        EEG.data is a string (the .fdt filename)
        Binary float32 stream of shape (n_times, n_channels), C-order

This is the layout used by all EEGLAB versions; ds005048 happens to use
case (1) — the data is embedded directly in the v7.3 HDF5 .set file.

Public API:
    read_raw_eeglab_v73(set_path) -> mne.io.RawArray
"""

from __future__ import annotations
import os
import numpy as np
import mne


def _scalar(x):
    """Coerce a 0-d/1-element pymatreader value to a Python scalar."""
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        return x.flatten()[0].item() if x.dtype.kind in "iuf" else x.flatten()[0]
    return x


def _string(x):
    """Coerce pymatreader's various string representations to a Python str."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray):
        if x.dtype.kind == "U":
            return "".join(x.flatten().tolist())
        if x.dtype.kind == "S":
            return b"".join(x.flatten().tolist()).decode("utf-8", errors="replace")
        if x.size == 1:
            return _string(x.flatten()[0])
        if x.dtype.kind in ("i", "u") and x.ndim <= 2:
            try:
                return "".join(chr(int(c)) for c in x.flatten() if int(c) > 0)
            except Exception:
                return str(x)
    return str(x)


def _ch_names_from_chanlocs(chanlocs, n_channels: int) -> list[str]:
    """
    pymatreader represents EEG.chanlocs as a dict-of-arrays (struct-of-arrays).
    The 'labels' field is an array (or list) of channel names.
    """
    if chanlocs is None:
        return [f"EEG{i+1:03d}" for i in range(n_channels)]

    if isinstance(chanlocs, dict) and "labels" in chanlocs:
        labels = chanlocs["labels"]
        if isinstance(labels, (list, tuple)):
            names = [_string(l) for l in labels]
        elif isinstance(labels, np.ndarray):
            names = [_string(l) for l in labels.flatten()]
        else:
            names = [_string(labels)]
    elif isinstance(chanlocs, (list, tuple)):
        names = []
        for entry in chanlocs:
            if isinstance(entry, dict) and "labels" in entry:
                names.append(_string(entry["labels"]))
            else:
                names.append("")
    else:
        names = []

    # Pad / fix
    names = [n if n else f"EEG{i+1:03d}" for i, n in enumerate(names)]
    if len(names) < n_channels:
        names += [f"EEG{i+1:03d}" for i in range(len(names), n_channels)]
    elif len(names) > n_channels:
        names = names[:n_channels]
    return names


def _orient_data(data: np.ndarray, n_channels_hint: int | None = None) -> np.ndarray:
    """Return data in (n_channels, n_times) orientation."""
    if data.ndim != 2:
        raise ValueError(f"Expected 2D EEG data, got shape {data.shape}")
    r, c = data.shape
    if n_channels_hint is not None:
        if r == n_channels_hint and c != n_channels_hint:
            return data
        if c == n_channels_hint and r != n_channels_hint:
            return data.T
    # Heuristic: more samples than channels
    return data if r < c else data.T


def read_raw_eeglab_v73(set_path: str, preload: bool = True) -> mne.io.RawArray:
    """
    Load a v7.3 (HDF5) EEGLAB .set file and return an mne.io.RawArray.

    Falls back to MNE's native reader for non-v7.3 files when possible.
    Parameters
    ----------
    set_path : str
        Path to the .set file.
    preload : bool
        Kept for API compatibility with mne.io.read_raw_eeglab.

    Returns
    -------
    raw : mne.io.RawArray
    """
    try:
        import pymatreader
    except ImportError as e:
        raise ImportError(
            "Loading v7.3 EEGLAB files requires pymatreader. "
            "Install with: pip install pymatreader"
        ) from e

    mat = pymatreader.read_mat(set_path)

    # The struct can be at top level or under key 'EEG'
    if "EEG" in mat:
        eeg = mat["EEG"]
    else:
        # When the .set is saved as the EEG struct directly at top level,
        # pymatreader exposes the fields directly.
        eeg = mat

    if not isinstance(eeg, dict):
        raise ValueError(f"Unexpected EEG struct type in {set_path}: {type(eeg)}")

    # --- Sampling rate --------------------------------------------------------
    sfreq = float(_scalar(eeg.get("srate")))
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError(f"Invalid sampling rate in {set_path}: {sfreq}")

    # --- Channel count --------------------------------------------------------
    nbchan_raw = _scalar(eeg.get("nbchan"))
    nbchan = int(nbchan_raw) if nbchan_raw is not None else None

    # --- Data array -----------------------------------------------------------
    data_field = eeg.get("data")

    if isinstance(data_field, np.ndarray) and data_field.dtype.kind in "fiu":
        # Embedded numerical array
        data = np.asarray(data_field, dtype=np.float64)
        data = _orient_data(data, nbchan)
    else:
        # data field is a filename pointing to a .fdt sibling
        fdt_name = _string(data_field)
        if not fdt_name:
            raise ValueError(f"Cannot determine data location in {set_path}")
        fdt_path = fdt_name
        if not os.path.isabs(fdt_path):
            fdt_path = os.path.join(os.path.dirname(set_path), fdt_name)
        if not os.path.exists(fdt_path):
            raise FileNotFoundError(f".fdt data file not found: {fdt_path}")

        n_pnts_raw = _scalar(eeg.get("pnts"))
        n_trials_raw = _scalar(eeg.get("trials"))
        n_pnts = int(n_pnts_raw) if n_pnts_raw is not None else None
        n_trials = int(n_trials_raw) if n_trials_raw is not None else 1

        if nbchan is None or n_pnts is None:
            raise ValueError(f"Missing nbchan/pnts in {set_path}; cannot read .fdt")

        flat = np.fromfile(fdt_path, dtype=np.float32)
        expected = nbchan * n_pnts * max(n_trials, 1)
        if flat.size != expected:
            raise ValueError(
                f".fdt size mismatch for {fdt_path}: "
                f"got {flat.size} samples, expected {expected} "
                f"(nbchan={nbchan}, pnts={n_pnts}, trials={n_trials})"
            )
        # EEGLAB binary layout is (nbchan, pnts*trials) in Fortran order
        data = flat.reshape((nbchan, n_pnts * max(n_trials, 1)), order="F").astype(np.float64)

    # EEGLAB stores data in microvolts; MNE expects volts in RawArray
    data = data * 1e-6

    # --- Channel names --------------------------------------------------------
    chanlocs = eeg.get("chanlocs")
    ch_names = _ch_names_from_chanlocs(chanlocs, data.shape[0])

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


def read_raw_eeglab_any(set_path: str, preload: bool = True) -> mne.io.RawArray:
    """
    Try MNE's native reader first; if it fails on v7.3, fall back to the
    pymatreader-based loader. Returns a RawArray either way.
    """
    try:
        return mne.io.read_raw_eeglab(set_path, preload=preload, verbose=False)
    except Exception as e:
        msg = str(e).lower()
        if "v7.3" in msg or "hdf" in msg or "h5py" in msg:
            return read_raw_eeglab_v73(set_path, preload=preload)
        raise
