#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug-Skript für Latent Shape-Analyse in AllTalk TTS

Dieses Skript untersucht die Dimensionen und Strukturen von Latent-Tensoren,
insbesondere für Speaker Embeddings. Es hilft bei der Identifikation von:
- Tensor-Dimensionen
- Möglichen Normalisierungsstrategien
- Strukturellen Unterschieden zwischen Latent-Dateien
"""

import json
import torch
import numpy as np

def check_latent_shape(latent_path):
    """
    Analysiert die Shape und Dimensionen eines Latent-Tensors.
    
    Args:
        latent_path (str): Pfad zur JSON-Latent-Datei
    """
    try:
        # JSON-Datei laden
        with open(latent_path, 'r') as f:
            latent_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Fehler beim Laden der Datei {latent_path}: {e}")
        return
    
    try:
        # Konvertiere speaker_embedding zu Tensor mit Fehlerbehandlung
        speaker_embedding = torch.tensor(latent_data['speaker_embedding'])
    except KeyError:
        print(f"Warnung: Kein 'speaker_embedding' in {latent_path} gefunden!")
        return
    except Exception as e:
        print(f"Fehler bei Tensor-Konvertierung: {e}")
        return
    
    # Detaillierte Shape-Analyse
    print(f"\nLatent Datei: {latent_path}")
    print(f"Ursprüngliche speaker_embedding Shape: {speaker_embedding.shape}")
    print(f"Ursprüngliche Dimensionen: {len(speaker_embedding.shape)}")
    
    # Dimensionen-Normalisierungsstrategien
    normalized_embeddings = []
    
    # Strategie 1: Squeeze
    if len(speaker_embedding.shape) == 3:
        embedding_squeezed = speaker_embedding.squeeze(0)
        normalized_embeddings.append(('squeeze(0)', embedding_squeezed))
    
    # Strategie 2: Flatten
    embedding_flattened = speaker_embedding.view(-1)
    normalized_embeddings.append(('flatten/view(-1)', embedding_flattened))
    
    # Strategie 3: Reshape basierend auf Dimensionen
    if len(speaker_embedding.shape) > 2:
        embedding_reshaped = speaker_embedding.reshape(-1)
        normalized_embeddings.append(('reshape(-1)', embedding_reshaped))
    
    # Ausgabe der normalisierten Embeddings
    for strategy, embedding in normalized_embeddings:
        print(f"\nNormalisierungsstrategie: {strategy}")
        print(f"Normalisierte Shape: {embedding.shape}")
        print(f"Normalisierte Dimensionen: {len(embedding.shape)}")
        print(f"Erste 5 Werte: {embedding[:5]}")

def main():
    """
    Hauptfunktion zum Ausführen der Latent Shape-Analyse.
    """
    # Pfade zu den zu analysierenden Latent-Dateien
    test_paths = [
        '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/Tom5.json',
        '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/merge_Tom5_male_01_male_01_male_01.json'
    ]
    
    # Analyse für jede Datei durchführen
    for path in test_paths:
        check_latent_shape(path)

if __name__ == "__main__":
    main()
