"""
model.py
────────
Builds and loads the EEGPT model configured for Alzheimer's classification.
Imported by train.py — do not run directly.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn as nn

# Add EEGPT repo to Python path so we can import from it
sys.path.append(str(Path.home() / "EEGPT"))
from downstream.Modules.models.EEGPT_mcae_finetune import EEGPTClassifier

# ── Channel configuration ─────────────────────────────────────────────────────
# Must exactly match the channels output by preprocess.py

USE_CHANNELS = [
    'FP1', 'FP2',
    'F7',  'F3', 'FZ', 'F4', 'F8',
    'T7',
    'C3',  'CZ', 'C4',
    'T8',
    'P7',
    'P3',  'PZ', 'P4',
    'P8',
    'O1',  'O2',
]


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(num_classes=2, checkpoint_path=None, device='cpu'):
    """
    Build EEGPTClassifier configured for your dataset and optionally
    load pretrained weights.

    Args:
        num_classes     : number of output classes (2 = AD vs Control)
        checkpoint_path : path to .ckpt pretrained weights, or None
        device          : 'cuda' or 'cpu'

    Returns:
        model ready for training
    """

    model = EEGPTClassifier(
        # Output size
        num_classes        = num_classes,

        # Input configuration
        in_channels        = 19,          # your 19 EEG channels
        img_size           = [19, 1024],  # [channels, timepoints per epoch]
        patch_stride       = 64,          # how far the patch window slides

        # Channel names — used to look up learned spatial embeddings
        use_channels_names = USE_CHANNELS,

        # Channel projection: maps your 19ch → EEGPT's internal space
        use_chan_conv      = True,

        # Freeze the pretrained encoder — only the head will train
        use_freeze_encoder = True,

        # Use predictor module (better for classification than reconstructor)
        use_predictor      = True,

        # Skip internal resampling since our epochs are already 1024 points
        desired_time_len   = 1024,
        use_avg            = False,
    )

    # ── Load pretrained weights ───────────────────────────────────────────────
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Download from EEGPT repo and place at that path."
            )

        # In model.py, replace the load_state_dict block with this:

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint  = torch.load(checkpoint_path, map_location='cpu')
        state_dict  = checkpoint.get('state_dict', checkpoint)
        
        # Replace the entire remapped = {} loop with this:
        remapped = {}
        for k, v in state_dict.items():
            # Step 1: top-level module rename
            if k.startswith('encoder.'):
                k = k.replace('encoder.', 'target_encoder.', 1)
            elif k.startswith('reconstructor.'):
                k = k.replace('reconstructor.', 'predictor.', 1)
        
            # Step 2: internal layer rename (reconstructor → predictor naming)
            k = k.replace('predictor.reconstructor_blocks', 'predictor.predictor_blocks')
            k = k.replace('predictor.reconstructor_embed',  'predictor.predictor_embed')
            k = k.replace('predictor.reconstructor_norm',   'predictor.predictor_norm')
            k = k.replace('predictor.reconstructor_proj',   'predictor.predictor_proj')
            k = k.replace('predictor.chan_embed',            'predictor.chan_embed')
        
            remapped[k] = v
        
        # Fix mask_token shape: (1,1,512) → (1,1,4,512)
        if 'predictor.mask_token' in remapped:
            mt = remapped['predictor.mask_token']
            remapped['predictor.mask_token'] = mt.unsqueeze(2).repeat(1, 1, 4, 1)

        
        # Add this right before the load_state_dict call in model.py
        # Remove mismatched projection layers so they initialize fresh
        remapped.pop('predictor.predictor_proj.weight', None)
        remapped.pop('predictor.predictor_proj.bias',   None)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        print(f"Missing keys    : {missing}")
        print(f"Unexpected keys : {unexpected}")

    # ── Verify freeze ─────────────────────────────────────────────────────────
    frozen   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFrozen params   : {frozen:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"(Only the classification head should be trainable)\n")

    model = model.to(device).float()
    # Force all Conv1d and Linear layers to stay in float32
    # by removing the autocast decorator effect
    
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            module.float()
    
    # Disable autocast inside the model by patching it
    from torch.cuda.amp import autocast
    _original_autocast = autocast.__init__
    def _disabled_autocast(self, *args, **kwargs):
        kwargs['enabled'] = False
        _original_autocast(self, *args, **kwargs)
    autocast.__init__ = _disabled_autocast
    
    return model
