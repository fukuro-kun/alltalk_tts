"""
Unittest-Modul für die Generierung von Merge-Dateinamen in AllTalk TTS.

Dieses Modul enthält Testfälle zur Überprüfung der Funktionalität 
zur Generierung von Dateinamen beim Mergen von Latent-Dateien.

Hauptfunktionen:
- Testen der Dateinamen-Generierung mit verschiedenen Eingabeformaten
- Überprüfen der Korrektheit und Konsistenz der Namenskonventionen
- Validierung der Merge-Dateinamen-Generierung
"""

import os
import sys
import unittest

# Pfad zum Hauptverzeichnis hinzufügen, um Module zu importieren
sys.path.append('/media/fukuro/raid5/alltalk_tts')

from stimmen import generate_merge_filename


class TestGenerateMergeFilename(unittest.TestCase):
    """
    Testklasse für die Generierung von Merge-Dateinamen.

    Diese Klasse enthält Testmethoden zur Überprüfung verschiedener 
    Aspekte der Dateinamen-Generierung beim Mergen von Latent-Dateien.
    """

    def test_generate_merge_filename_with_full_paths(self):
        """
        Test mit vollständigen Pfaden zu JSON-Dateien
        """
        sources = {
            'gpt_cond_latent': [
                {'json': '/path/to/male_01.json', 'name': 'male_01'},
                {'json': '/path/to/male_02.json', 'name': 'male_02'}
            ],
            'speaker_embedding': [
                {'json': '/path/to/Tom5.json', 'name': 'Tom5'}
            ]
        }
        expected_filename = 'merge_male_01_male_02_Tom5.json'
        result = generate_merge_filename(sources)
        self.assertEqual(result, expected_filename)

    def test_generate_merge_filename_with_filenames(self):
        """
        Test mit reinen Dateinamen
        """
        sources = {
            'gpt_cond_latent': [
                {'json': 'male_01.json', 'name': 'male_01'},
                {'json': 'male_02.json', 'name': 'male_02'}
            ],
            'speaker_embedding': [
                {'json': 'Tom5.json', 'name': 'Tom5'}
            ]
        }
        expected_filename = 'merge_male_01_male_02_Tom5.json'
        result = generate_merge_filename(sources)
        self.assertEqual(result, expected_filename)

    def test_generate_merge_filename_with_dict_sources(self):
        """
        Test mit bereits gemergten Latent-Dictionaries
        """
        sources = {
            'gpt_cond_latent': [
                {'json': {'data': 'some_data'}, 'name': 'merged_gpt1'},
                {'json': {'data': 'other_data'}, 'name': 'merged_gpt2'}
            ],
            'speaker_embedding': [
                {'json': {'data': 'speaker_data'}, 'name': 'Tom5'}
            ]
        }
        expected_filename = 'merge_merged_gpt1_merged_gpt2_Tom5.json'
        result = generate_merge_filename(sources)
        self.assertEqual(result, expected_filename)

    def test_generate_merge_filename_without_name(self):
        """
        Test mit Quellen ohne expliziten Namen
        """
        sources = {
            'gpt_cond_latent': [
                {'json': '/path/to/male_01.json'},
                {'json': '/path/to/male_02.json'}
            ],
            'speaker_embedding': [
                {'json': '/path/to/Tom5.json'}
            ]
        }
        expected_filename = 'merge_male_01_male_02_Tom5.json'
        result = generate_merge_filename(sources)
        self.assertEqual(result, expected_filename)

    def test_generate_merge_filename_empty_sources(self):
        """
        Test mit leeren Quellen
        """
        sources = {}
        with self.assertRaises(ValueError):
            generate_merge_filename(sources)

    def test_generate_merge_filename_partial_sources(self):
        """
        Test mit teilweise leeren Quellen
        """
        sources = {
            'gpt_cond_latent': [],
            'speaker_embedding': [
                {'json': 'Tom5.json', 'name': 'Tom5'}
            ]
        }
        expected_filename = 'merge_Tom5.json'
        result = generate_merge_filename(sources)
        self.assertEqual(result, expected_filename)


if __name__ == '__main__':
    unittest.main()
