import json
import torch
import numpy as np

def check_latent_shape(latent_path):
    with open(latent_path, 'r') as f:
        latent_data = json.load(f)
    
    # Konvertiere speaker_embedding zu Tensor
    speaker_embedding = torch.tensor(latent_data['speaker_embedding'])
    
    print(f"Latent Datei: {latent_path}")
    print(f"speaker_embedding Shape: {speaker_embedding.shape}")
    print(f"speaker_embedding Dimensionen: {len(speaker_embedding.shape)}")
    
    # Versuche, die Dimensionen zu normalisieren
    if len(speaker_embedding.shape) == 3:
        speaker_embedding = speaker_embedding.squeeze(0)
    
    print(f"Normalisierte Shape: {speaker_embedding.shape}")
    print(f"Normalisierte Dimensionen: {len(speaker_embedding.shape)}")

# Teste beide Dateien
check_latent_shape('/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/Tom5.json')
check_latent_shape('/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/merge_Tom5_male_01_male_01_male_01.json')
