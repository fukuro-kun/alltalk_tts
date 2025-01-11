import json
import numpy as np
import torch

def debug_merge_process(latent_paths):
    # Lade Latent-Dateien
    latents = [json.load(open(path, 'r')) for path in latent_paths]
    
    print("Ursprüngliche Latent-Strukturen:")
    for i, latent in enumerate(latents):
        print(f"Latent {i}:")
        print("gpt_cond_latent Shape:", np.array(latent['gpt_cond_latent']).shape)
        print("speaker_embedding Shape:", np.array(latent['speaker_embedding']).shape)
        print("speaker_embedding als Tensor:", torch.tensor(latent['speaker_embedding']).shape)
        print()
    
    # Merge-Simulation
    def merge_latents(latents, weights):
        merged_gpt = np.zeros_like(latents[0]['gpt_cond_latent'])
        merged_speaker = np.zeros_like(latents[0]['speaker_embedding'])
        
        for latent, weight in zip(latents, weights):
            merged_gpt += np.array(latent['gpt_cond_latent']) * weight
            merged_speaker += np.array(latent['speaker_embedding']) * weight
        
        return {
            'gpt_cond_latent': merged_gpt,
            'speaker_embedding': merged_speaker
        }
    
    # Teste mit gleichen Gewichten
    weights = [1/len(latents)] * len(latents)
    merged_latent = merge_latents(latents, weights)
    
    print("Gemergter Latent:")
    print("gpt_cond_latent Shape:", np.array(merged_latent['gpt_cond_latent']).shape)
    print("speaker_embedding Shape:", np.array(merged_latent['speaker_embedding']).shape)
    print("speaker_embedding als Tensor:", torch.tensor(merged_latent['speaker_embedding']).shape)

# Teste mit den beiden Latent-Dateien
debug_merge_process([
    '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/Tom5.json',
    '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/male_01.json'
])
