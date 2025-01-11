#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug-Skript für Latent-Dimensionen-Merge in AllTalk TTS

Dieses Skript analysiert den Merge-Prozess von Latent-Tensoren,
insbesondere für GPT Conditioning und Speaker Embeddings. Es hilft bei:
- Verständnis der Dimensionszusammenführung
- Gewichteten Merge-Strategien
- Detaillierter Schritt-für-Schritt-Analyse
"""

import json
import numpy as np
import os

def detailed_merge_debug(latent_paths):
    """
    Führt eine detaillierte Analyse des Latent-Merge-Prozesses durch.
    
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
            gpt_latent = np.array(latent['gpt_cond_latent'])
            speaker_embedding = np.array(latent['speaker_embedding'])
            
            print("gpt_cond_latent Shape:", gpt_latent.shape)
            print("gpt_cond_latent Datentyp:", gpt_latent.dtype)
            print("gpt_cond_latent Erste 5 Werte:", gpt_latent[:5])
            
            print("speaker_embedding Shape:", speaker_embedding.shape)
            print("speaker_embedding Datentyp:", speaker_embedding.dtype)
            print("speaker_embedding Erste 5 Werte:", speaker_embedding[:5])
        
        except KeyError as e:
            print(f"Fehler: Schlüssel {e} nicht gefunden!")
        except Exception as e:
            print(f"Unerwarteter Fehler: {e}")
    
    def merge_latents(latents, weights):
        """
        Führt einen gewichteten Merge der Latent-Tensoren durch.
        
        Args:
            latents (list): Liste der Latent-Dictionaries
            weights (list): Gewichtungsfaktoren für jeden Latent
        
        Returns:
            dict: Gemergter Latent mit gpt_cond_latent und speaker_embedding
        """
        # Initialisiere Zielstrukturen mit gleicher Form wie Eingabe
        merged_gpt = np.zeros_like(latents[0]['gpt_cond_latent'])
        merged_speaker = np.zeros_like(latents[0]['speaker_embedding'])
        
        print("\nMerge-Prozess Schritt für Schritt:")
        for i, (latent, weight) in enumerate(zip(latents, weights)):
            print(f"\nSchritt {i+1}:")
            print(f"Aktueller Latent: {os.path.basename(latent_paths[i])}")
            print(f"Gewichtung: {weight}")
            
            # Zeige Eingabewerte vor dem Mergen
            gpt_input = np.array(latent['gpt_cond_latent'])
            speaker_input = np.array(latent['speaker_embedding'])
            
            print("Eingabe gpt_cond_latent Shape:", gpt_input.shape)
            print("Eingabe speaker_embedding Shape:", speaker_input.shape)
            
            # Merge-Vorgang mit Dimensionsprüfung
            try:
                merged_gpt += gpt_input * weight
                merged_speaker += speaker_input * weight
            except ValueError as e:
                print(f"Dimensionsfehler beim Mergen: {e}")
                return None
            
            # Zeige Zwischenergebnis
            print("Aktuelles Zwischen-Ergebnis merged_speaker:")
            print(merged_speaker)
        
        return {
            'gpt_cond_latent': merged_gpt,
            'speaker_embedding': merged_speaker
        }
    
    # Teste mit verschiedenen Gewichtungsstrategien
    weight_strategies = [
        [1/len(latents)] * len(latents),  # Gleichmäßige Verteilung
        [0.6, 0.4],  # Ungleiche Gewichtung
    ]
    
    for strategy_index, weights in enumerate(weight_strategies):
        print(f"\n--- Gewichtungsstrategie {strategy_index + 1} ---")
        print("Gewichte:", weights)
        
        merged_latent = merge_latents(latents, weights)
        
        if merged_latent:
            print("\nEndgültiger Gemergter Latent:")
            print("gpt_cond_latent Shape:", np.array(merged_latent['gpt_cond_latent']).shape)
            print("speaker_embedding Shape:", np.array(merged_latent['speaker_embedding']).shape)
            print("speaker_embedding Detailansicht:")
            print(np.array(merged_latent['speaker_embedding']))

def main():
    """
    Hauptfunktion zum Ausführen der Latent-Merge-Analyse.
    """
    # Pfade zu den zu analysierenden Latent-Dateien
    test_paths = [
        '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/Tom5.json',
        '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/male_01.json'
    ]
    
    detailed_merge_debug(test_paths)

if __name__ == "__main__":
    main()
