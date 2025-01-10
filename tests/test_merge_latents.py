"""
Detaillierte Analyse der Latent-JSON-Struktur in XTTS-Modellen

Struktur der Latent-Dateien (z.B. male_01.json, Tom5.json):

1. Hauptstruktur:
   - Ein JSON-Objekt mit zwei Hauptschlüsseln: 'gpt_cond_latent' und 'speaker_embedding'
   - Beide Schlüssel haben eine nahezu identische verschachtelte Listenstruktur

2. 'gpt_cond_latent' Detailstruktur:
   - Erste Ebene: Liste mit GENAU EINEM Element
     Beispiel: [[ ... ]]
   
   - Zweite Ebene: Liste mit GENAU 32 Elementen
     Jedes Element ist eine Liste von 1024 Floating-Point-Werten
     Beispiel: [
         [1.632, -0.031, -0.152, ..., weitere Werte],   # 1024 Werte
         [4.0, 5.0, 6.0, ..., weitere Werte]            # Weitere 1024 Werte
         ...
     ]

   - Dritte Ebene: 1024 Floating-Point-Werte pro Element
     - Präzise: float32/float64
     - Wertebereich: Typischerweise zwischen -2.0 und 2.0
     - Hohe numerische Varianz und Genauigkeit

3. 'speaker_embedding' Detailstruktur:
   - Erste Ebene: Liste mit GENAU EINEM Element
   - Zweite Ebene: Liste mit GENAU 512 Elementen
   - Dritte Ebene: Liste mit GENAU EINEM Floating-Point-Wert
     Beispiel: [
         [[0.009735506]],
         [[0.046235539]],
         ...
     ]

4. Dimensionen und Charakteristiken:
   - 'gpt_cond_latent': Struktur [1][32][1024]
   - 'speaker_embedding': Struktur [1][512][1]
   - Konsistente Verschachtelung über verschiedene Latent-Dateien

Wichtige Beobachtungen:
- Strikte, vorhersehbare Verschachtelungsstruktur
- Nahezu identische Listenstruktur für beide Latent-Typen
- Floating-Point-Werte mit hoher Präzision und numerischer Varianz

Dieses Modul enthält Testfälle zur Überprüfung der Merge-Funktion für Latent-Dateien.
Es testet verschiedene Szenarien wie Merging mit gleichen und ungleichen Gewichtungen
sowie die Kompatibilitätsprüfung von Latent-Dateien.

Hauptfunktionen:
- Testen des Mergens mit gleichen Gewichtungen
- Testen des Mergens mit unterschiedlichen Gewichtungen
- Überprüfen der Kompatibilität von Latent-Dateien
"""

import os
import sys
import json
import unittest
import numpy as np

# Pfad zum Hauptverzeichnis hinzufügen, um Module zu importieren
sys.path.append('/media/fukuro/raid5/alltalk_tts')

from stimmen import merge_latents, check_latent_compatibility


class TestMergeLatents(unittest.TestCase):
    """
    Testklasse für die Latent-Merge-Funktionalität.

    Diese Klasse enthält Testmethoden zur Überprüfung verschiedener Aspekte
    der Latent-Merge-Funktion, einschließlich Gewichtung und Strukturerhaltung.
    """

    def setUp(self):
        """
        Initialisiert Testdaten vor jedem Testfall.

        Erstellt Beispiel-Latent-Dictionaries mit definierter Struktur:
        - gpt_cond_latent: [1][32][1024]
        - speaker_embedding: [1][512][1]

        Die Beispieldaten werden für verschiedene Merge-Szenarien verwendet.
        """
        self.sample_latent1 = {
            'gpt_cond_latent': [
                [
                    [1.0] * 1024 for _ in range(32)
                ]
            ],
            'speaker_embedding': [
                [
                    [[0.1]] for _ in range(512)
                ]
            ]
        }
        
        self.sample_latent2 = {
            'gpt_cond_latent': [
                [
                    [10.0] * 1024 for _ in range(32)
                ]
            ],
            'speaker_embedding': [
                [
                    [[1.0]] for _ in range(512)
                ]
            ]
        }

    def test_merge_latents_equal_weights(self):
        """
        Testet das Mergen von Latents mit gleichen Gewichtungen (50/50).

        Überprüft:
        - Korrekte Merge-Struktur
        - Richtiger Mittelwert bei gleichen Gewichtungen
        - Beibehaltung der ursprünglichen Listenstruktur
        """
        sources = {
            'gpt_cond_latent': [
                {'json': self.sample_latent1, 'weight': 0.5},
                {'json': self.sample_latent2, 'weight': 0.5}
            ],
            'speaker_embedding': [
                {'json': self.sample_latent1, 'weight': 0.5},
                {'json': self.sample_latent2, 'weight': 0.5}
            ]
        }
        
        # Führe Merge durch
        merged_latent = merge_latents(sources)
        
        # Überprüfe Struktur
        self.assertEqual(len(merged_latent['gpt_cond_latent']), 1)  # Eine Liste
        self.assertEqual(len(merged_latent['gpt_cond_latent'][0]), 32)  # 32 Elemente
        self.assertEqual(len(merged_latent['gpt_cond_latent'][0][0]), 1024)  # 1024 Werte pro Element
        
        # Überprüfe erste Elemente
        np.testing.assert_almost_equal(
            merged_latent['gpt_cond_latent'][0][0],
            [5.5] * 1024,  # (1.0 + 10.0)/2
            decimal=7
        )

    def test_merge_latents_unequal_weights(self):
        """
        Testet das Mergen von Latents mit unterschiedlichen Gewichtungen.

        Überprüft:
        - Korrekte Merge-Struktur
        - Richtiger gewichteter Durchschnitt
        - Korrekte Berücksichtigung der Gewichtungen
        """
        sources = {
            'gpt_cond_latent': [
                {'json': self.sample_latent1, 'weight': 0.3},
                {'json': self.sample_latent2, 'weight': 0.7}
            ],
            'speaker_embedding': [
                {'json': self.sample_latent1, 'weight': 0.6},
                {'json': self.sample_latent2, 'weight': 0.4}
            ]
        }
        
        # Führe Merge durch
        merged_latent = merge_latents(sources)
        
        # Überprüfe Struktur
        self.assertEqual(len(merged_latent['gpt_cond_latent']), 1)  # Eine Liste
        self.assertEqual(len(merged_latent['gpt_cond_latent'][0]), 32)  # 32 Elemente
        self.assertEqual(len(merged_latent['gpt_cond_latent'][0][0]), 1024)  # 1024 Werte pro Element
        
        # Überprüfe erste Elemente mit gewichteten Durchschnitten
        np.testing.assert_almost_equal(
            merged_latent['gpt_cond_latent'][0][0],
            [7.3] * 1024,  # 1.0 * 0.3 + 10.0 * 0.7
            decimal=7
        )

    def test_merge_latents_incompatible(self):
        """
        Testet die Kompatibilitätsprüfung von Latent-Dateien.

        Überprüft:
        - Erkennung von Latents mit unterschiedlichen Strukturen
        - Korrekte Auslösung einer ValueError
        """
        incompatible_latent = {
            'gpt_cond_latent': [[[1.0, 2.0]]],  # Andere Dimensionen
            'speaker_embedding': [[[0.1], [0.2]]]
        }
        
        sources = {
            'gpt_cond_latent': [
                {'json': self.sample_latent1, 'weight': 0.5},
                {'json': incompatible_latent, 'weight': 0.5}
            ],
            'speaker_embedding': [
                {'json': self.sample_latent1, 'weight': 0.5},
                {'json': incompatible_latent, 'weight': 0.5}
            ]
        }
        
        # Prüfe, dass Kompatibilitätsprüfung fehlschlägt
        with self.assertRaises(ValueError):
            check_latent_compatibility([s['json'] for s in sources['gpt_cond_latent'] + sources['speaker_embedding']])


if __name__ == '__main__':
    unittest.main()
