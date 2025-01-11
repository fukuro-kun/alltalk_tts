import os
import json
import torch
import pytest
import numpy as np

# Importiere Funktionen aus stimmen.py
from stimmen import get_latent_directory

def load_latent_from_json(latent_name: str) -> dict:
    """
    Lädt einen Latent aus einer JSON-Datei.
    
    Args:
        latent_name (str): Name des Latent-JSONs
    
    Returns:
        dict: Geladener Latent als Dictionary
    """
    latent_path = os.path.join(get_latent_directory(), f"{latent_name}.json")
    with open(latent_path, 'r') as f:
        return json.load(f)

def calculate_latent_similarity(latent1: dict, latent2: dict) -> dict:
    """
    Berechnet die Ähnlichkeit zwischen zwei Latent-Dictionaries.
    
    Args:
        latent1 (dict): Erster Latent als Dictionary
        latent2 (dict): Zweiter Latent als Dictionary
    
    Returns:
        dict: Ähnlichkeitswerte für verschiedene Latent-Komponenten
    """
    def calculate_component_similarity(component1: list, component2: list) -> float:
        """
        Berechnet Ähnlichkeit für eine Latent-Komponente.
        
        Args:
            component1 (list): Erste Latent-Komponente
            component2 (list): Zweite Latent-Komponente
        
        Returns:
            float: Ähnlichkeitswert als Prozentsatz
        """
        # Konvertiere zu Tensoren
        latent1_tensor = torch.tensor(component1)
        latent2_tensor = torch.tensor(component2)
        
        # Flache Tensoren, falls mehrdimensional
        latent1_flat = latent1_tensor.view(-1)
        latent2_flat = latent2_tensor.view(-1)
        
        # Stelle sicher, dass beide Tensoren gleiche Länge haben
        min_length = min(len(latent1_flat), len(latent2_flat))
        latent1_flat = latent1_flat[:min_length]
        latent2_flat = latent2_flat[:min_length]
        
        # Normalisiere die Latents
        latent1_norm = torch.nn.functional.normalize(latent1_flat, p=2, dim=0)
        latent2_norm = torch.nn.functional.normalize(latent2_flat, p=2, dim=0)
        
        # Kosinus-Ähnlichkeit
        cosine_similarity = torch.nn.functional.cosine_similarity(
            latent1_norm.unsqueeze(0), 
            latent2_norm.unsqueeze(0), 
            dim=1
        )
        
        # Konvertiere zu Prozentsatz
        similarity_percentage = cosine_similarity.item() * 100
        
        return similarity_percentage
    
    # Berechne Ähnlichkeiten für verschiedene Komponenten
    gpt_similarity = calculate_component_similarity(
        latent1['gpt_cond_latent'], 
        latent2['gpt_cond_latent']
    )
    
    speaker_similarity = calculate_component_similarity(
        latent1['speaker_embedding'], 
        latent2['speaker_embedding']
    )
    
    # Gesamtähnlichkeit als gewichteter Durchschnitt
    # GPT Latent hat mehr Elemente, daher mehr Gewicht
    total_similarity = (
        (gpt_similarity * 0.7) + 
        (speaker_similarity * 0.3)
    )
    
    return {
        'total_similarity': total_similarity,
        'gpt_similarity': gpt_similarity,
        'speaker_similarity': speaker_similarity
    }

def test_latent_similarity_same_latent():
    """
    Testet die Ähnlichkeitsberechnung mit identischen Latents.
    Erwartet: Sehr hohe Ähnlichkeit (nahe 100%)
    """
    # Wähle einen Latent zum Testen
    latent_name = "male_02"  # Beispiel-Latent
    
    # Lade Latent
    latent = load_latent_from_json(latent_name)
    
    # Berechne Ähnlichkeit mit sich selbst
    similarity = calculate_latent_similarity(latent, latent)
    
    print(f"Selbst-Ähnlichkeit für {latent_name}:")
    print(f"Gesamtähnlichkeit: {similarity['total_similarity']:.6f}%")
    print(f"GPT Ähnlichkeit: {similarity['gpt_similarity']:.6f}%")
    print(f"Speaker Ähnlichkeit: {similarity['speaker_similarity']:.6f}%")
    
    # Erwarte sehr hohe Ähnlichkeit (99.9% oder höher)
    assert similarity['total_similarity'] > 99.9, f"Selbst-Ähnlichkeit zu niedrig: {similarity['total_similarity']}%"

def test_latent_similarity_different_latents():
    """
    Testet die Ähnlichkeitsberechnung zwischen verschiedenen Latents.
    Erwartet: Geringere Ähnlichkeit
    """
    # Wähle zwei verschiedene Latents
    latent1_name = "male_02"
    latent2_name = "male_01"
    
    # Lade Latents
    latent1 = load_latent_from_json(latent1_name)
    latent2 = load_latent_from_json(latent2_name)
    
    # Berechne Ähnlichkeit
    similarity = calculate_latent_similarity(latent1, latent2)
    
    print(f"Ähnlichkeit zwischen {latent1_name} und {latent2_name}:")
    print(f"Gesamtähnlichkeit: {similarity['total_similarity']:.6f}%")
    print(f"GPT Ähnlichkeit: {similarity['gpt_similarity']:.6f}%")
    print(f"Speaker Ähnlichkeit: {similarity['speaker_similarity']:.6f}%")
    
    # Erwarte geringere Ähnlichkeit (unter 90%)
    assert similarity['total_similarity'] < 90, f"Ähnlichkeit zwischen verschiedenen Latents zu hoch: {similarity['total_similarity']}%"

def test_latent_similarity_speaker_embeddings():
    """
    Testet die Ähnlichkeitsberechnung für Speaker Embeddings.
    """
    # Wähle zwei Speaker Latents
    latent1_name = "male_02"
    latent2_name = "male_01"
    
    # Lade Speaker Embeddings
    latent1 = load_latent_from_json(latent1_name)
    latent2 = load_latent_from_json(latent2_name)
    
    # Berechne Ähnlichkeit
    similarity = calculate_latent_similarity(latent1, latent2)
    
    print(f"Speaker Embedding Ähnlichkeit zwischen {latent1_name} und {latent2_name}:")
    print(f"Gesamtähnlichkeit: {similarity['total_similarity']:.6f}%")
    print(f"GPT Ähnlichkeit: {similarity['gpt_similarity']:.6f}%")
    print(f"Speaker Ähnlichkeit: {similarity['speaker_similarity']:.6f}%")
    
    # Erwarte Ähnlichkeit zwischen 0 und 90%
    assert 0 <= similarity['total_similarity'] < 90, f"Unerwartete Speaker Embedding Ähnlichkeit: {similarity['total_similarity']}%"

def test_latent_similarity_tensor_shapes():
    """
    Testet die Ähnlichkeitsberechnung mit unterschiedlichen Tensorformen.
    """
    # Erzeuge Beispiel-Tensoren
    latent1 = torch.rand(32768)  # GPT Conditioning Latent Form
    latent2 = torch.rand(32768)
    
    # Berechne Ähnlichkeit
    similarity = calculate_latent_similarity(
        {
            'gpt_cond_latent': latent1.tolist(), 
            'speaker_embedding': torch.rand(512).tolist()
        }, 
        {
            'gpt_cond_latent': latent2.tolist(), 
            'speaker_embedding': torch.rand(512).tolist()
        }
    )
    
    print(f"Ähnlichkeit zufälliger Tensoren:")
    print(f"Gesamtähnlichkeit: {similarity['total_similarity']:.6f}%")
    print(f"GPT Ähnlichkeit: {similarity['gpt_similarity']:.6f}%")
    print(f"Speaker Ähnlichkeit: {similarity['speaker_similarity']:.6f}%")
    
    # Erwarte Ähnlichkeit zwischen 0 und 90%
    assert 0 <= similarity['total_similarity'] < 90, f"Unerwartete Tensor-Ähnlichkeit: {similarity['total_similarity']}%"

# Optional: Hauptausführung für manuelle Tests
if __name__ == "__main__":
    pytest.main([__file__])
