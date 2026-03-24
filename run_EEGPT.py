class EEGAlzheimersPipeline(nn.Module):
    def __init__(self, use_channels_names):
        super().__init__()
        
        # Frozen EEGPT encoder
        self.eegpt = EEGPTClassifier(
            num_classes=0,          # no head, returns embeddings
            in_channels=19,
            img_size=[19, 1024],
            use_channels_names=use_channels_names,
            use_chan_conv=True,
            use_freeze_encoder=True,
            use_predictor=True,
        )
        
        # Simple trainable classifier on top
        self.ad_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)       # AD vs non-AD
        )
    
    def forward(self, x):
        with torch.no_grad():
            embeddings = self.eegpt.forward_features(x)  # (B, 512)
        return self.ad_head(embeddings)                   # (B, 2)