#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug-Skript für JSON-Konvertierung in AllTalk TTS

Dieses Skript analysiert die Konvertierung von JSON-Dateien mit Latent-Daten,
insbesondere für Speaker Embeddings. Es hilft bei der Untersuchung von:
- Datenstruktur
- Tensor-Konvertierung
- Shape-Analyse
"""

import json
import numpy as np
import torch

def detailed_json_debug(input_paths):
    """
    Führt eine detaillierte Analyse der JSON-Konvertierung durch.
    
    Args:
        input_paths (list): Liste der JSON-Dateipfade zur Analyse
    """
    print("Detaillierte JSON-Konversions-Analyse:")
    
    for path in input_paths:
        print(f"\nDatei: {path}")
        
        # JSON laden und Fehlerbehandlung
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Fehler beim Laden der Datei: {e}")
            continue
        
        # Numpy-Konvertierung mit Fehlerbehandlung
        try:
            np_speaker_embedding = np.array(data['speaker_embedding'])
            print("Numpy Array direkt aus JSON:")
            print("Shape:", np_speaker_embedding.shape)
            print("Datentyp:", np_speaker_embedding.dtype)
            print("Erste Werte:", np_speaker_embedding[0, :5] if np_speaker_embedding.ndim > 1 else np_speaker_embedding[:5])
        except KeyError:
            print("Warnung: Kein 'speaker_embedding' in der JSON-Datei gefunden!")
        
        # Torch-Konvertierung mit Fehlerbehandlung
        try:
            torch_speaker_embedding = torch.tensor(data['speaker_embedding'])
            print("\nTorch Tensor:")
            print("Shape:", torch_speaker_embedding.shape)
            print("Datentyp:", torch_speaker_embedding.dtype)
            print("Erste Werte:", torch_speaker_embedding[0, :5] if torch_speaker_embedding.ndim > 1 else torch_speaker_embedding[:5])
        except KeyError:
            print("Warnung: Kein 'speaker_embedding' für Torch-Konvertierung gefunden!")
        
        # Detaillierte Shape-Analyse
        print("\nShape-Analyse:")
        for i, dim_size in enumerate(np_speaker_embedding.shape):
            print(f"Dimension {i}: {dim_size}")
        
        # Zusätzliche Informationen
        print("\nZusätzliche Informationen:")
        print("Verfügbare Schlüssel:", list(data.keys()))

def main():
    """
    Hauptfunktion zum Ausführen der JSON-Debug-Analyse.
    """
    # Pfade zu den zu analysierenden JSON-Dateien
    test_paths = [
        '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/Tom5.json',
        '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents/merge_Tom5_male_01_male_01_male_01.json'
    ]
    
    detailed_json_debug(test_paths)

if __name__ == "__main__":
    main()
