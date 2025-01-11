import json
import numpy as np

def detailed_merge_debug(latent_paths):
    # Lade Latent-Dateien
    latents = [json.load(open(path, 'r')) for path in latent_paths]
    
    print("Ursprüngliche Latent-Strukturen:")
    for i, latent in enumerate(latents):
        print(f"Latent {i}:")
        print("gpt_cond_latent Shape:", np.array(latent['gpt_cond_latent']).shape)
        print("speaker_embedding Shape:", np.array(latent['speaker_embedding']).shape)
        print("speaker_embedding Detailansicht:")
        print(np.array(latent['speaker_embedding']))
        print()
    
    # Merge-Simulation mit expliziter Schrittanzeige
    def merge_latents(latents, weights):
        # Initialisiere Zielstrukturen mit gleicher Form wie Eingabe
        merged_gpt = np.zeros_like(latents[0]['gpt_cond_latent'])
        merged_speaker = np.zeros_like(latents[0]['speaker_embedding'])
        
        print("Merge-Prozess Schritt für Schritt:")
        for i, (latent, weight) in enumerate(zip(latents, weights)):
            print(f"\nSchritt {i+1}:")
            print(f"Aktueller Latent: {latent_paths[i]}")
            print(f"Gewichtung: {weight}")
            
            # Zeige Eingabewerte vor dem Mergen
            print("Eingabe gpt_cond_latent:", np.array(latent['gpt_cond_latent']))
            print("Eingabe speaker_embedding:", np.array(latent['speaker_embedding']))
            
            # Merge-Vorgang
            merged_gpt += np.array(latent['gpt_cond_latent']) * weight
            merged_speaker += np.array(latent['speaker_embedding']) * weight
            
            # Zeige Zwischenergebnis
            print("Aktuelles Zwischen-Ergebnis speaker_embedding:")
            print(merged_speaker)
        
        return {
            'gpt_cond_latent': merged_gpt,
            'speaker_embedding': merged_speaker
        }
    
    # Teste mit gleichen Gewichten
    weights = [1/len(latents)] * len(latents)
    merged_latent = merge_latents(latents, weights)
    
    print("\nEndgültiger Gemergter Latent:")
    print("gpt_cond_latent Shape:", np.array(merged_latent['gpt_cond_latent']).shape)
    print("speaker_embedding Shape:", np.array(merged_latent['speaker_embedding']).shape)
    print("speaker_embedding Detailansicht:")
    print(np.array(merged_latent['speaker_embedding']))

# Teste mit den beiden Latent-Dateien
detailed_merge_debug([
    '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/Tom5.json',
    '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/male_01.json'
])
