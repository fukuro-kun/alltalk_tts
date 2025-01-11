#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug-Skript für Merge-Prozess in AllTalk TTS

Dieses Skript analysiert den gesamten Merge-Prozess von Latent-Tensoren,
mit Fokus auf:
- Detaillierte Schrittanalyse
- Verschiedene Merge-Strategien
- Tensor-Transformationen
- Vergleich von Numpy und Torch Repräsentationen
"""

import json
import numpy as np
import torch
import os

def debug_merge_process(latent_paths):
    """
    Führt eine detaillierte Analyse des Merge-Prozesses durch.
    
    Args:
        latent_paths (list): Liste der Pfade zu Latent-JSON-Dateien
    """
    # Lade Latent-Dateien mit Fehlerbehandlung
    latents = []
    for path in latent_paths:
        try:
            with open(path, 'r') as f:
                latent_data = json.load(f)
                latents.append(latent_data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Fehler beim Laden von {path}: {e}")
            return
    
    print("Ursprüngliche Latent-Strukturen:")
    for i, latent in enumerate(latents):
        try:
            print(f"\nLatent {i} ({os.path.basename(latent_paths[i])}):")
            
            # GPT Conditioning Latent Analyse
            gpt_cond_latent_np = np.array(latent['gpt_cond_latent'])
            gpt_cond_latent_torch = torch.tensor(latent['gpt_cond_latent'])
            
            print("gpt_cond_latent Numpy Shape:", gpt_cond_latent_np.shape)
            print("gpt_cond_latent Numpy Datentyp:", gpt_cond_latent_np.dtype)
            print("gpt_cond_latent Torch Shape:", gpt_cond_latent_torch.shape)
            print("gpt_cond_latent Torch Datentyp:", gpt_cond_latent_torch.dtype)
            
            # Speaker Embedding Analyse
            speaker_embedding_np = np.array(latent['speaker_embedding'])
            speaker_embedding_torch = torch.tensor(latent['speaker_embedding'])
            
            print("speaker_embedding Numpy Shape:", speaker_embedding_np.shape)
            print("speaker_embedding Numpy Datentyp:", speaker_embedding_np.dtype)
            print("speaker_embedding Torch Shape:", speaker_embedding_torch.shape)
            print("speaker_embedding Torch Datentyp:", speaker_embedding_torch.dtype)
        
        except KeyError as e:
            print(f"Fehler: Schlüssel {e} nicht gefunden!")
        except Exception as e:
            print(f"Unerwarteter Fehler: {e}")
    
    def merge_latents(latents, weights, merge_strategy='weighted_average'):
        """
        Führt einen Merge der Latent-Tensoren mit verschiedenen Strategien durch.
        
        Args:
            latents (list): Liste der Latent-Dictionaries
            weights (list): Gewichtungsfaktoren für jeden Latent
            merge_strategy (str): Merge-Strategie ('weighted_average', 'concatenate')
        
        Returns:
            dict: Gemergter Latent mit gpt_cond_latent und speaker_embedding
        """
        print(f"\nMerge-Strategie: {merge_strategy}")
        
        if merge_strategy == 'weighted_average':
            # Gewichteter Durchschnitt
            merged_gpt = np.zeros_like(latents[0]['gpt_cond_latent'])
            merged_speaker = np.zeros_like(latents[0]['speaker_embedding'])
            
            for latent, weight in zip(latents, weights):
                merged_gpt += np.array(latent['gpt_cond_latent']) * weight
                merged_speaker += np.array(latent['speaker_embedding']) * weight
        
        elif merge_strategy == 'concatenate':
            # Verkettung der Latents
            merged_gpt = np.concatenate([np.array(latent['gpt_cond_latent']) for latent in latents])
            merged_speaker = np.concatenate([np.array(latent['speaker_embedding']) for latent in latents])
        
        else:
            raise ValueError(f"Unbekannte Merge-Strategie: {merge_strategy}")
        
        return {
            'gpt_cond_latent': merged_gpt,
            'speaker_embedding': merged_speaker
        }
    
    # Verschiedene Gewichtungsstrategien
    weight_strategies = [
        ([1/len(latents)] * len(latents), 'weighted_average'),
        ([0.6, 0.4], 'weighted_average'),
        ([1, 1], 'concatenate')
    ]
    
    for weights, strategy in weight_strategies:
        print(f"\n--- Merge mit Strategie: {strategy} ---")
        print("Gewichte:", weights)
        
        try:
            merged_latent = merge_latents(latents, weights, strategy)
            
            print("\nGemergter Latent:")
            print("gpt_cond_latent Shape:", np.array(merged_latent['gpt_cond_latent']).shape)
            print("gpt_cond_latent als Torch Tensor Shape:", torch.tensor(merged_latent['gpt_cond_latent']).shape)
            
            print("speaker_embedding Shape:", np.array(merged_latent['speaker_embedding']).shape)
            print("speaker_embedding als Torch Tensor Shape:", torch.tensor(merged_latent['speaker_embedding']).shape)
        
        except Exception as e:
            print(f"Fehler beim Mergen: {e}")

def main():
    """
    Hauptfunktion zum Ausführen der Merge-Prozess-Analyse.
    """
    # Pfade zu den zu analysierenden Latent-Dateien
    test_paths = [
        '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/Tom5.json',
        '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/male_01.json'
    ]
    
    debug_merge_process(test_paths)

if __name__ == "__main__":
    main()
