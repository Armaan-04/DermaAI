

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from config import MODEL_NAME, NUM_CLASSES, FEATURE_SIZE


class SkinConditionClassifier(nn.Module):
    """MedSigLIP-based eight-class skin-condition classifier."""

    def __init__(self):
        super().__init__()

        # Load MedSigLIP configuration and correct text token IDs
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.text_config.bos_token_id = None
        config.text_config.eos_token_id = 1
        config.text_config.pad_token_id = 1

        # Load pretrained MedSigLIP model
        self.backbone = AutoModel.from_pretrained(
            MODEL_NAME,
            config=config,
            dtype=torch.float32
        )

        # Freeze the pretrained encoder initially
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Eight-class classification head
        self.classifier = nn.Linear(FEATURE_SIZE, NUM_CLASSES)

    def forward(self, pixel_values):
        # Extract image features without calculating encoder gradients
        with torch.no_grad():
            vision_outputs = self.backbone.vision_model(
                pixel_values=pixel_values
            )

        # Obtain pooled image representation
        features = vision_outputs.pooler_output

        # Predict condition scores
        logits = self.classifier(features)

        return logits



