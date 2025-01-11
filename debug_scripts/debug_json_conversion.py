import json
import numpy as np
import torch

def detailed_json_debug(input_paths):
    print("Detaillierte JSON-Konversions-Analyse:")
    
    for path in input_paths:
        print(f"\nDatei: {path}")
        
        # JSON laden
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Numpy-Konvertierung
        np_speaker_embedding = np.array(data['speaker_embedding'])
        print("Numpy Array direkt aus JSON:")
        print("Shape:", np_speaker_embedding.shape)
        print("Datentyp:", np_speaker_embedding.dtype)
        print("Erste Werte:", np_speaker_embedding[0, :5])
        
        # Torch-Konvertierung
        torch_speaker_embedding = torch.tensor(data['speaker_embedding'])
        print("\nTorch Tensor:")
        print("Shape:", torch_speaker_embedding.shape)
        print("Datentyp:", torch_speaker_embedding.dtype)
        print("Erste Werte:", torch_speaker_embedding[0, :5])
        
        # Detaillierte Shape-Analyse
        print("\nShape-Analyse:")
        for i, dim_size in enumerate(np_speaker_embedding.shape):
            print(f"Dimension {i}: {dim_size}")

# Teste mit beiden Dateien
detailed_json_debug([
    '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/Tom5.json',
    '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/merge_Tom5_male_01_male_01_male_01.json'
])
