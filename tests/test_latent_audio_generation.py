"""
Unittest-Modul für die Audio-Generierung aus Latent-Dateien.

Dieses Modul testet die Generierung von Vorschau-Audios 
aus verschiedenen Latent-JSON-Dateien.

Hauptfunktionen:
- Überprüfung der Audio-Generierung für verschiedene Latents
- Validierung der generierten Audiodateien
- Detaillierte Diagnose von Generierungsproblemen
"""

import os
import sys
import unittest
import librosa
import numpy as np
import pytest
import logging
import json
import traceback
import asyncio

# Importiere die korrekte Funktion
from stimmen import generate_latent_audio


class TestLatentAudioGeneration(unittest.TestCase):
    """
    Testklasse für die Audio-Generierung aus Latent-Dateien.

    Testet die Generierung von Vorschau-Audios und 
    liefert detaillierte Debugging-Informationen.
    """

    tts_engine = None  # Klassenattribut für TTS-Engine

    @classmethod
    def setUpClass(cls):
        # Stelle sicher, dass die TTS-Engine vor den Tests initialisiert wird
        from system.tts_engines.xtts.model_engine import tts_class as XTTSEngine
        cls.tts_engine = XTTSEngine()
        asyncio.run(cls.tts_engine.setup())

    def setUp(self):
        """
        Initialisiert Testumgebung und definiert Latent-Dateien.
        """
        self.latent_dir = '/media/fukuro/raid5/alltalk_tts/voices/xtts_latents'
        self.latent_files = [
            'male_01.json', 
            'Tom5.json', 
            'male_02.json'
        ]
        
        # Importiere TTS-Engine und lade Modell
        from system.tts_engines.xtts.model_engine import tts_class
        
        # Initialisiere TTS-Engine
        self.tts_engine = tts_class()
        
        # Scanne verfügbare Modelle
        self.tts_engine.scan_models_folder()
        
        # Versuche, das Modell zu laden
        try:
            # Verwende 'xtts' als Präfix und 'xttsv2_2.0.3' als Modellnamen
            print("🚀 Lade TTS-Modell für Unittest")
            # Verwende asyncio.run für asynchrone Methode
            asyncio.run(self.tts_engine.handle_tts_method_change("xtts - xttsv2_2.0.3"))
            print("✅ TTS-Modell erfolgreich geladen")
            
            # Zusätzliche Überprüfung des geladenen Modells
            print(f"🔍 Aktuell geladenes Modell: {self.tts_engine.current_model_loaded}")
            print(f"🔍 Modell-Instanz vorhanden: {self.tts_engine.model is not None}")
        except Exception as e:
            print(f"❌ Fehler beim Laden des TTS-Modells: {e}")
            traceback.print_exc()  # Detaillierte Fehlerausgabe
            raise

    def _debug_latent_file(self, latent_path):
        """
        Extrahiert und druckt detaillierte Informationen über die Latent-Datei.
        
        Args:
            latent_path (str): Pfad zur Latent-JSON-Datei
        """
        try:
            with open(latent_path, 'r') as f:
                latent_data = json.load(f)
            
            print("\n🔍 Latent-Datei Debugging:")
            print(f"Pfad: {latent_path}")
            
            # Überprüfe Schlüssel
            print("Vorhandene Schlüssel:", list(latent_data.keys()))
            
            # Strukturdetails
            for key in ['gpt_cond_latent', 'speaker_embedding']:
                if key in latent_data:
                    data = latent_data[key]
                    print(f"\n{key} Struktur:")
                    print(f"  Ebenen: {len(data)}")
                    print(f"  Erste Ebene Länge: {len(data[0])}")
                    
                    # Unterschiedliche Behandlung für verschiedene Latent-Typen
                    if key == 'gpt_cond_latent':
                        # Für gpt_cond_latent: Zeige nur Dimensionen
                        print(f"  Dimensionen: {len(data)}x{len(data[0])}x{len(data[0][0])}")
                        print(f"  Erste Werte (Stichprobe): {data[0][0][:5]} (von {len(data[0][0])} Elementen)")
                    elif key == 'speaker_embedding':
                        # Für speaker_embedding: Zeige kompakte Statistik
                        flat_data = np.array(data[0]).flatten()
                        print(f"  Dimensionen: {len(data)}x{len(data[0])}x{len(data[0][0])}")
                        print(f"  Erste Werte: {data[0][0][:3]}")
                    
                    # Wertebereich
                    flat_data = np.array(data[0]).flatten()
                    print(f"  Wertebereich: [{flat_data.min()}, {flat_data.max()}]")
                    print(f"  Mittelwert: {flat_data.mean()}")
                    print(f"  Standardabweichung: {flat_data.std()}")
        
        except Exception as e:
            print(f"❌ Fehler beim Debuggen der Latent-Datei: {e}")
            traceback.print_exc()

    def test_generate_preview_audio_for_latents(self):
        """
        Testet die Audio-Generierung für alle verfügbaren Latent-Dateien.

        Überprüft:
        - Erfolgreiche Audio-Generierung
        - Korrekte Dateierstellung
        - Mindestlänge der generierten Audiodatei
        - Liefert detaillierte Debugging-Informationen
        """
        for latent_file in self.latent_files:
            full_path = os.path.join(self.latent_dir, latent_file)
            
            # Debug-Informationen zur Latent-Datei
            self._debug_latent_file(full_path)
            
            with self.subTest(latent_file=latent_file):
                try:
                    # Generiere Vorschau-Audio
                    preview_audio_path = generate_latent_audio(os.path.splitext(latent_file)[0])
                    
                    # Detaillierte Ausgabe
                    print(f"\n📢 Audio-Generierung für {latent_file}:")
                    print(f"Ausgabepfad: {preview_audio_path}")
                    
                    # Überprüfe Datei-Existenz
                    if not os.path.exists(preview_audio_path):
                        print(f"❌ Keine Audiodatei für {latent_file} generiert!")
                        self.fail(f"Keine Audiodatei für {latent_file} generiert")
                    
                    # Dateigröße
                    file_size = os.path.getsize(preview_audio_path)
                    print(f"Dateigröße: {file_size} Bytes")
                    
                    # Überprüfe Dateigröße
                    self.assertGreater(
                        file_size, 1000, 
                        f"Audio für {latent_file} zu klein"
                    )
                    
                    # Audio-Validierung mit librosa
                    audio, sr = librosa.load(preview_audio_path)
                    
                    # Audiodetails
                    print(f"Sampling Rate: {sr} Hz")
                    print(f"Audiodauer: {len(audio)/sr:.2f} Sekunden")
                    
                    # Überprüfe Audiodauer
                    self.assertGreater(
                        len(audio) / sr, 1.0, 
                        f"Audio für {latent_file} zu kurz"
                    )
                    
                    print(f"✅ Audio für {latent_file} erfolgreich generiert")
                
                except Exception as e:
                    # Detaillierte Fehlerausgabe
                    print(f"❌ Schwerwiegender Fehler bei {latent_file}:")
                    print(f"Fehlertyp: {type(e).__name__}")
                    print(f"Fehlermeldung: {str(e)}")
                    traceback.print_exc()
                    raise

    def test_latent_path_handling(self):
        """
        Testet die korrekte Pfadbehandlung für Latent-Dateien.
        
        Überprüft:
        - Existenz der Latent-Datei
        - Erfolgreiche Latent-Ladung
        - Korrekte Dimensionen der Latents
        """
        # Importiere die TTS-Engine
        from system.tts_engines.xtts.model_engine import tts_class
        from stimmen import get_latent_directory
        import os
        import logging

        # Konfiguriere Logging
        logging.basicConfig(level=logging.DEBUG, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        # Initialisiere TTS-Engine
        tts_engine = tts_class()

        # Wähle einen vorhandenen Latent
        latent_name = "male_01"
        latent_dir = get_latent_directory()
        latent_path = os.path.join(latent_dir, f"{latent_name}.json")

        # Überprüfe Existenz der Latent-Datei
        assert os.path.exists(latent_path), f"Latent-Datei nicht gefunden: {latent_path}"

        # Teste Pfadbehandlung
        try:
            gpt_cond_latent, speaker_embedding = tts_engine._load_latents(f"latent:{latent_name}.json")
            logger.info(f"✅ Latent {latent_name} erfolgreich geladen")
            logger.info(f"GPT Cond Latent Shape: {gpt_cond_latent.shape}")
            logger.info(f"Speaker Embedding Shape: {speaker_embedding.shape}")
            
            # Zusätzliche Validierungen
            assert gpt_cond_latent is not None, "GPT Cond Latent darf nicht None sein"
            assert speaker_embedding is not None, "Speaker Embedding darf nicht None sein"
            assert gpt_cond_latent.dim() > 0, "GPT Cond Latent muss mindestens eine Dimension haben"
            assert speaker_embedding.dim() > 0, "Speaker Embedding muss mindestens eine Dimension haben"
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden des Latents: {e}")
            raise


def test_generate_latent_audio():
    """
    Testet die Audio-Generierung aus Latent-Dateien mit umfangreichem Logging.
    """
    # Nutze die Klasseninstanz der TTS-Engine
    tts_engine = TestLatentAudioGeneration.tts_engine

    # Konfiguriere Logging mit Stream-Handler
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout),  # Ausgabe auf Konsole
                            logging.StreamHandler(sys.stderr)   # Fehlerausgabe
                        ])
    logger = logging.getLogger(__name__)

    # Liste der zu testenden Latent-Namen
    test_latents = [
        "male_01",  # Ein Beispiel-Latent
        "male_02"   # Ein weiteres Beispiel-Latent
    ]

    for latent_name in test_latents:
        logger.info(f"🔍 Teste Audio-Generierung für Latent: {latent_name}")
        
        # Generiere Audio
        output_path = generate_latent_audio(os.path.splitext(latent_name)[0])
        
        # Überprüfe Ergebnisse
        assert output_path is not None, f"❌ Audio-Generierung für {latent_name} fehlgeschlagen"
        assert os.path.exists(output_path), f"❌ Audiodatei nicht gefunden: {output_path}"
        assert os.path.getsize(output_path) > 0, f"❌ Audiodatei ist leer: {output_path}"
        
        logger.info(f"✅ Audio-Generierung für {latent_name} erfolgreich: {output_path}")

def test_generate_latent_audio_with_custom_text():
    """
    Testet die Audio-Generierung mit benutzerdefiniertem Text.
    """
    # Nutze die Klasseninstanz der TTS-Engine
    tts_engine = TestLatentAudioGeneration.tts_engine

    # Konfiguriere Logging mit Stream-Handler
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout),  # Ausgabe auf Konsole
                            logging.StreamHandler(sys.stderr)   # Fehlerausgabe
                        ])
    logger = logging.getLogger(__name__)

    custom_text = "Heute ist ein wunderschöner Tag für Experimente und Innovationen."
    latent_name = "male_01"

    logger.info(f"🔍 Teste Audio-Generierung mit benutzerdefiniertem Text für {latent_name}")
    
    # Generiere Audio mit benutzerdefiniertem Text
    output_path = generate_latent_audio(os.path.splitext(latent_name)[0], text=custom_text)
    
    # Überprüfe Ergebnisse
    assert output_path is not None, f"❌ Audio-Generierung für {latent_name} mit benutzerdefiniertem Text fehlgeschlagen"
    assert os.path.exists(output_path), f"❌ Audiodatei nicht gefunden: {output_path}"
    assert os.path.getsize(output_path) > 0, f"❌ Audiodatei ist leer: {output_path}"
    
    logger.info(f"✅ Audio-Generierung mit benutzerdefiniertem Text für {latent_name} erfolgreich: {output_path}")

def test_generate_latent_audio_with_invalid_latent():
    """
    Testet die Fehlerbehandlung bei ungültigem Latent-Namen.
    """
    # Konfiguriere Logging mit Stream-Handler
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout),  # Ausgabe auf Konsole
                            logging.StreamHandler(sys.stderr)   # Fehlerausgabe
                        ])
    logger = logging.getLogger(__name__)

    invalid_latent_names = [
        "nicht_existierender_latent",
        "",
        None
    ]

    for invalid_name in invalid_latent_names:
        logger.info(f"🔍 Teste Fehlerbehandlung für ungültigen Latent-Namen: {invalid_name}")
        
        # Generiere Audio mit ungültigem Latent
        output_path = generate_latent_audio(invalid_name)
        
        # Überprüfe Ergebnisse
        assert output_path is None, f"❌ Ungültiger Latent-Name {invalid_name} sollte None zurückgeben"
        
        logger.info(f"✅ Fehlerbehandlung für ungültigen Latent-Namen {invalid_name} erfolgreich")

if __name__ == '__main__':
    unittest.main()
