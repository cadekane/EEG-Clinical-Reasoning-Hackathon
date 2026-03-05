# EEG-Clinical-Reasoning-Hackathon

We are tasked with building a language-based reasoning system to perform on raw EEG data. The system will be focused on Alzheimer's diagnosis and text-based clinical reasoning, i.e. not simply a classification task. 

Goals:
- A working system (not a single model)
- Clinician-style reasoning in language (claims + evidence)
- Handling uncertainty and confounds (artifacts, drowsiness, noise)
- Clear recommended next step(s)

Input:
- Raw EEG data
- Task prompts that require reasoning (not just classification)
- An evaluation based on a set of test cases or challenges will be provided

The AI system must analyze raw EEG data, then generate and compare multiple
clinical hypotheses:
• Extract signal features from raw EEG
• Map features → interpretable patterns
• Generate differential diagnoses
• Compare hypotheses using evidence weighting
• Output structured clinical reasoning text
• Explain evidence-for and evidence-against each hypothesis
• Track uncertainty (confidence + what would change your mind)

Here, biomarkers are not hand-crafted — they are learned from data.

**Pipeline**
Raw EEG → CNN / Transformer / GNN → Latent embedding space → Learned discriminative features
Latent features → mapped to interpretable biomarkers

