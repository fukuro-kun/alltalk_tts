"""
AllTalk TTS - Stimmen Management Modul

Dieses Modul bietet Funktionen für:
- Latent-Verzeichnis-Management
- Modell-Pfadkonstruktion
- Latent-Merge-Prozesse
- Audio-Generierung

Hauptanwendungsbereich: Text-to-Speech Latent-Verarbeitung

Hauptfunktionen Funktionen in Reihenfolge:

## Basis-Utility-Funktionen
1. get_latent_directory():
   - Ermittelt das Basis-Verzeichnis für Latents
   - Unterstützt flexible Fallback-Optionen
   - Erstellt Verzeichnis bei Bedarf
   - Logging-Unterstützung

2. get_model_directory():
   - Bestimmt Pfad für spezifische TTS-Modellversionen
   - Erstellt Modellverzeichnis, falls nicht vorhanden
   - Standardisierte Modellpfad-Konstruktion

3. get_model_files():
   - Identifiziert wichtige Modelldateien
   - Prüft Existenz von Konfigurationen, Checkpoints, Vokabular
   - Flexible Pfadermittlung

## Such- und Filterfunktionen
4. find_best_models():
   - Sucht nach 'best_model.pth' Dateien
   - Optional: Filterung nach Sprecher-Namen

5. find_jsons():
   - Findet spezifische JSON-Dateien
   - Optionale Filterung nach Sprecher-Namen

## Latent-Management-Kernfunktionen
6. load_available_latents():
   - Lädt verfügbare Latent-JSONs aus verschiedenen Verzeichnissen
    
   - Dynamische Verzeichnissuche
   - Fehler- und Logging-Behandlung
   - Entfernt Duplikate und sortiert Ergebnisse

7. check_latent_compatibility():
   - Prüft Dimensionalität von Latents
   - Verhindert Merge von inkompatiblen Latents

8. merge_latents():
   - Führt Latents mit gewichteter Methode zusammen
   - Normalisiert Gewichtungen
   - Unterstützt GPT Conditioning und Speaker Embedding

9. generate_merge_filename():
   - Generiert eindeutigen Dateinamen für gemergten Latent
   - Basiert auf Quell-Latent-Namen

## Audio-Generierung
10. generate_latent_audio():
    - Generiert Audio aus Latent
    - Unterstützt benutzerdefinierte Texte
    - Fehlerbehandlung und Debugging
    - Temporäre Datei-Generierung

## UI-Komponenten
11. setup_stimmen_tab():
    - Erstellt komplexe Gradio-Benutzeroberfläche
    - Integriert Latent-Management-Funktionen
    - Implementiert Slider-Synchronisation
    - Handhabt Modell-Laden und Audio-Generierung

12. initialize_xtts_model():
    - Initialisiert das XTTS-Modell für Latent-Generierung
    - Lädt das Modell aus Konfigurations- und Checkpoint-Dateien
    - Verwendet die Vokabulardatei für Tokenisierung
    - Verwendet DeepSpeed für Acceleration, falls verfügbar
    - Fügt ein Speaker Embedding Layer hinzu, falls vorhanden
    - Verwendet den GPU-Bereich, wenn verfügbar
    - Gibt das initialisierte Modell zurück

## Latent-Ähnlichkeitsberechnung
13. calculate_latent_similarity():
    - Berechnet die Ähnlichkeit zwischen zwei Latent-Tensoren
    - Verwendet Kosinus-Ähnlichkeit

14. setup_similarity_buttons():
    - Richtet Buttons und Ausgabefelder für Latent-Ähnlichkeitsberechnung ein
    - Integriert Funktionen für GPT und Speaker Latent-Ähnlichkeitsberechnung

"""

import os
import sys
import json
import numpy as np
import gradio as gr
import glob
import logging
import time
from functools import wraps
import tempfile
import traceback
import torch
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
import torchaudio

def get_latent_directory(base_dir: str | None = None) -> str:
    """
    Ermittelt das Basis-Verzeichnis für Latents mit flexiblen Fallback-Optionen.

    Args:
        base_dir (str, optional): Basis-Verzeichnis für Latent-Suche.
            Defaults to None.

    Returns:
        str: Vollständiger Pfad zum Latent-Verzeichnis.

    Raises:
        OSError: Wenn kein geeignetes Verzeichnis gefunden werden kann.
    """
    # Logging-Konfiguration
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Verzeichnis-Ermittlung mit Fallback-Optionen
    if base_dir is None:
        potential_dirs = [
            os.environ.get('ALLTALK_BASE_DIR'),
            '/media/fukuro/raid5/alltalk_tts',
            os.path.dirname(__file__)
        ]
        
        for potential_dir in potential_dirs:
            if potential_dir and os.path.exists(potential_dir):
                base_dir = potential_dir
                break
        
        if base_dir is None:
            raise ValueError("Kein gültiges Basis-Verzeichnis gefunden")
    
    latent_dir = os.path.join(base_dir, "voices", "xtts_latents")
    
    # Erstelle Verzeichnis, falls nicht vorhanden
    try:
        os.makedirs(latent_dir, exist_ok=True)
        logger.info(f"Latent-Verzeichnis festgelegt: {latent_dir}")
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Latent-Verzeichnisses: {e}")
        raise
    
    return latent_dir

def get_model_directory(model_name: str = "xtts", version: str = "2.0.3") -> str:
    """
    Ermittelt das Verzeichnis für ein spezifisches TTS-Modell.

    Args:
        model_name (str, optional): Name des TTS-Modells. 
            Defaults to "xtts".
        version (str, optional): Version des Modells. 
            Defaults to "2.0.3".

    Returns:
        str: Vollständiger Pfad zum Modellverzeichnis.

    Raises:
        OSError: Wenn das Modellverzeichnis nicht erstellt werden kann.
    """
    model_path = f"/media/fukuro/raid5/alltalk_tts/models/{model_name}/xttsv2_{version}"
    
    if not os.path.exists(model_path):
        logging.warning(f"Modellverzeichnis nicht gefunden: {model_path}")
        os.makedirs(model_path, exist_ok=True)
    
    return model_path

def get_model_files(model_path: str | None = None) -> dict[str, str]:
    """
    Ermittelt die Pfade zu wichtigen Modelldateien.

    Args:
        model_path (str, optional): Pfad zum Modellverzeichnis. 
            Defaults to None.

    Returns:
        dict: Pfade zu verschiedenen Modelldateien mit Schlüsseln:
            - 'config': Pfad zur Konfigurationsdatei
            - 'model': Pfad zur Modell-Checkpoint-Datei
            - 'vocab': Pfad zur Vokabular-Datei

    Raises:
        FileNotFoundError: Wenn erforderliche Modelldateien fehlen.
    """
    if model_path is None:
        model_path = get_model_directory()
    
    model_files = {
        'config': os.path.join(model_path, 'config.json'),
        'checkpoint': os.path.join(model_path, 'model.pth'),
        'vocab': os.path.join(model_path, 'vocab.json'),
        'speakers': os.path.join(model_path, 'speakers_xtts.pth')
    }
    
    # Überprüfe Existenz der Dateien
    for name, filepath in model_files.items():
        if not os.path.exists(filepath):
            logging.warning(f"{name.capitalize()} Datei nicht gefunden: {filepath}")
    
    return model_files

def find_best_models(directory: str, speaker_name: str | None = None) -> list[str]:
    """
    Sucht nach 'best_model.pth' Dateien in einem gegebenen Verzeichnis.

    Args:
        directory (str): Verzeichnispfad zur Suche nach Modellen.
        speaker_name (str, optional): Optionaler Sprecher-Name zum Filtern. 
            Defaults to None.

    Returns:
        list[str]: Liste der gefundenen Modell-Pfade.

    Raises:
        ValueError: Wenn das Verzeichnis nicht existiert.
    """
    best_models = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'best_model.pth':
                # Optional: Filtere nach Speaker Name, wenn angegeben
                if speaker_name is None or speaker_name in root:
                    best_models.append(os.path.join(root, file))
    return best_models

def find_jsons(directory: str, filename: str, speaker_name: str | None = None) -> list[str]:
    """
    Sucht nach spezifischen JSON-Dateien in einem gegebenen Verzeichnis.

    Args:
        directory (str): Verzeichnispfad zur Suche nach JSON-Dateien.
        filename (str): Name der zu suchenden JSON-Datei.
        speaker_name (str, optional): Optionaler Sprecher-Name zum Filtern. 
            Defaults to None.

    Returns:
        list[str]: Liste der gefundenen JSON-Dateipfade.

    Raises:
        ValueError: Wenn das Verzeichnis nicht existiert.
    """
    json_files = []
    for root, dirs, files in os.walk(directory):
        if filename in files:
            # Optional: Filtere nach Speaker Name, wenn angegeben
            if speaker_name is None or speaker_name in root:
                json_files.append(os.path.join(root, filename))
    return json_files

def load_available_latents(directories: str | list[str] | None = None) -> list[str]:
    """
    Lädt verfügbare Latent-JSONs aus verschiedenen Verzeichnissen.

    Args:
        directories (str | list[str], optional): Verzeichnis(se) zur Suche 
            nach Latent-JSONs. Defaults to None.

    Returns:
        list[str]: Liste der verfügbaren Latent-Dateinamen.

    Raises:
        ValueError: Wenn keine Verzeichnisse gefunden werden.
        IOError: Bei Problemen beim Lesen der Verzeichnisse.
    """
    # Logging-Konfiguration
    logger = logging.getLogger(__name__)
    
    # Wenn keine Verzeichnisse angegeben, nutze Standardverzeichnis
    if directories is None:
        directories = [get_latent_directory()]
    
    # Stelle sicher, dass directories eine Liste ist
    if isinstance(directories, str):
        directories = [directories]
    
    available_latents = []
    for directory in directories:
        try:
            # Überprüfe Verzeichnis-Existenz
            if not os.path.exists(directory):
                logger.warning(f"Verzeichnis nicht gefunden: {directory}")
                continue
            
            # Suche JSON-Dateien
            latent_files = [
                f for f in os.listdir(directory) 
                if f.endswith('.json') and not f.startswith('.')
            ]
            
            # Erweitere Liste mit vollständigen Pfaden
            available_latents.extend([
                os.path.splitext(f)[0] for f in latent_files
            ])
            
            logger.info(f"Gefundene Latents in {directory}: {latent_files}")
        
        except Exception as e:
            logger.error(f"Fehler beim Suchen von Latents in {directory}: {e}")
    
    # Entferne Duplikate und sortiere
    return sorted(set(available_latents))

def check_latent_compatibility(latent_jsons: list[dict]) -> bool:
    """
    Prüft die Dimensionalität und Kompatibilität von Latent-JSONs.

    Args:
        latent_jsons (list[dict]): Liste der Latent-JSON-Objekte zur Überprüfung.

    Returns:
        bool: True, wenn alle Latents kompatibel sind, sonst False.

    Raises:
        ValueError: Bei ungültigen Latent-Strukturen.
    """
    lengths = {
        'gpt_cond_latent': [len(json['gpt_cond_latent'][0][0]) for json in latent_jsons],
        'speaker_embedding': [len(json['speaker_embedding'][0]) for json in latent_jsons]
    }
    
    if len(set(lengths['gpt_cond_latent'])) > 1 or len(set(lengths['speaker_embedding'])) > 1:
        raise ValueError("Latents haben unterschiedliche Dimensionen und können nicht gemergt werden.")
    
    return True

def merge_latents(latent_sources: dict[str, list[dict]]) -> dict:
    """
    Führt Latents mit gewichteter Methode zusammen.

    Args:
        latent_sources (dict[str, list[dict]]): Dictionary mit Latent-Quellen, 
            getrennt nach 'gpt_cond_latent' und 'speaker_embedding'.
            Jede Quelle enthält ein JSON und eine Gewichtung.

    Returns:
        dict: Zusammengeführtes Latent-JSON mit gemittelten Werten.
        Struktur bleibt identisch zum Eingabe-JSON.

    Raises:
        ValueError: Bei ungültigen Latent-Strukturen oder Gewichtungen.
    """
    # Prüfe Eingabedaten
    if not latent_sources or not all(key in latent_sources for key in ['gpt_cond_latent', 'speaker_embedding']):
        raise ValueError("Ungültige Latent-Quellen")

    # Initialisiere Ergebnis-Dictionary
    merged_latent = {}

    # Merge für jeden Latent-Typ
    for latent_type in ['gpt_cond_latent', 'speaker_embedding']:
        sources = latent_sources[latent_type]
        
        # Normalisiere Gewichtungen
        total_weight = sum(source['weight'] for source in sources)
        normalized_sources = [
            {'json': source['json'], 'normalized_weight': source['weight'] / total_weight} 
            for source in sources
        ]

        # Extrahiere Latent-Daten
        latent_data = [source['json'][latent_type][0] for source in normalized_sources]

        # Merge-Logik mit Beibehaltung der Originalstruktur
        if latent_type == 'gpt_cond_latent':
            # Struktur: [1][32][1024]
            merged_data = [[
                [
                    np.average(
                        [source[j][i] for source in latent_data], 
                        weights=[src['normalized_weight'] for src in normalized_sources],
                        axis=0
                    ) 
                    for i in range(len(latent_data[0][0]))
                ] 
                for j in range(len(latent_data[0]))
            ]]
        else:  # speaker_embedding
            # Struktur: [1][512][1]
            merged_data = [[
                [np.average(
                    [source[j][0] for source in latent_data], 
                    weights=[src['normalized_weight'] for src in normalized_sources],
                    axis=0
                ).tolist()]  # Konvertiere NumPy-Array zu Liste
                for j in range(len(latent_data[0]))
            ]]

        merged_latent[latent_type] = merged_data

    return merged_latent

def generate_merge_filename(sources: dict[str, list[dict]]) -> str:
    """
    Generiert einen eindeutigen Dateinamen für gemergten Latent.

    Args:
        sources (dict[str, list[dict]]): Dictionary mit Latent-Quellen, 
            getrennt nach 'gpt_cond_latent' und 'speaker_embedding'.

    Returns:
        str: Generierter Dateiname im Format 'merge_[Quellname1]_[Quellname2].json'.

    Raises:
        ValueError: Wenn keine Quellen vorhanden sind.
    """
    if not sources:
        raise ValueError("Keine Latent-Quellen zum Mergen gefunden")
    
    def extract_name(source: dict) -> str:
        """
        Extrahiert den Namen aus einer Latent-Quelle.
        
        Args:
            source (dict): Dictionary mit 'json' Schlüssel
        
        Returns:
            str: Name des Latents
        """
        # Wenn 'json' bereits ein Dictionary ist (gemergter Latent), extrahiere Namen
        if isinstance(source['json'], dict):
            return source.get('name', 'unknown')
        
        # Wenn 'json' ein Pfad oder Dateiname ist
        json_path = source['json']
        # Versuche, den Namen aus dem Dateinamen zu extrahieren
        name = os.path.splitext(os.path.basename(json_path))[0]
        return name if name else 'unknown'
    
    gpt_sources = [extract_name(s) for s in sources.get('gpt_cond_latent', [])]
    speaker_sources = [extract_name(s) for s in sources.get('speaker_embedding', [])]
    
    # Generiere Dateinamen
    merge_sources = gpt_sources + speaker_sources
    merge_filename = f"merge_{'_'.join(merge_sources)}.json"
    
    return merge_filename

def generate_latent_audio(
    latent_name: str, 
    text: str | None = None, 
    output_dir: str | None = None,
    model_name: str = "xtts",
    version: str = "2.0.3"
) -> str:
    """
    Generiert Audio aus einem beliebigen Latent.

    Args:
        latent_name (str): Name des Latents (ohne .json-Erweiterung).
        text (str, optional): Text zur Audio-Generierung. 
            Verwendet Standardtext, wenn None. Defaults to None.
        output_dir (str, optional): Verzeichnis für generierte Audios. 
            Erstellt temporäres Verzeichnis, wenn None. Defaults to None.
        model_name (str, optional): Name des TTS-Modells. Defaults to "xtts".
        version (str, optional): Version des Modells. Defaults to "2.0.3".

    Returns:
        str: Pfad zur generierten Audiodatei.
    """
    # Statische Modell-Initialisierung mit bestehender Funktion
    if not hasattr(generate_latent_audio, 'model'):
        generate_latent_audio.model = initialize_xtts_model()
    
    # Standard-Text für Vorschau
    default_text = "Dies ist eine Latent Audio Vorschau."
    
    # Latent-Datei laden
    latent_path = os.path.join(get_latent_directory(), f"{latent_name}.json")
    with open(latent_path, 'r') as f:
        latent_data = json.load(f)
    
    # Tensor-Generierung
    gpt_cond_latent = torch.tensor(latent_data['gpt_cond_latent'])
    speaker_embedding = torch.tensor(latent_data['speaker_embedding'])
    
    # Audio-Generierung
    out = generate_latent_audio.model.inference(
        text=text or default_text,
        language="de",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7,
        length_penalty=1.0,
        repetition_penalty=2.0
    )
    
    # Audio speichern
    output_dir = output_dir or tempfile.mkdtemp()
    output_path = os.path.join(output_dir, f"{latent_name}_preview.wav")
    
    out_wav = torch.tensor(out["wav"]).unsqueeze(0)
    torchaudio.save(output_path, out_wav, 24000)
    
    return output_path

def prepare_latent_name(latent_name: str) -> str:
    """
    Bereitet den Latent-Namen für die Audio-Generierung vor.
    Entfernt .json-Erweiterung, falls vorhanden.
    
    Args:
        latent_name (str): Ursprünglicher Latent-Name
    
    Returns:
        str: Bereinigter Latent-Name
    """
    return os.path.splitext(latent_name)[0]

def setup_stimmen_tab(demo: gr.Blocks) -> gr.Blocks:
    """
    Erstellt den Stimmen Management Tab mit Latent Merge Funktionalität.

    Diese Funktion konfiguriert eine komplexe Gradio-Benutzeroberfläche für:
    - Latent-Verwaltung
    - Audio-Generierung
    - Latent-Merging

    Args:
        demo (gr.Blocks): Bestehende Gradio Blocks-Instanz zur Erweiterung.

    Returns:
        gr.Blocks: Aktualisierte Gradio Blocks-Instanz mit Stimmen-Tab.

    Raises:
        ImportError: Wenn erforderliche Bibliotheken nicht geladen werden können.
        ValueError: Bei Konfigurationsproblemen der Benutzeroberfläche.
    """
    # Modellpfad dynamisch ermitteln
    xtts_model_path = get_model_directory()
    
    # Standardtext für Generierungen
    default_text = "Der alte Meister saß in seinem Studierzimmer und betrachtete die alten Schriftrollen. Die Zeit schien stillzustehen, während er die vergilbten Seiten durchblätterte! Draußen tobte ein Sturm, aber hier drinnen war es warm und gemütlich."
    
    # Latent-Verzeichnis
    latent_dir = get_latent_directory()
    os.makedirs(latent_dir, exist_ok=True)
    
    # 1. ZUERST: Dropdowns schnell befüllen
    available_latents = load_available_latents()
    
    # 2. Modell-Initialisierung im Hintergrund starten
    def background_model_init():
        try:
            demo.xtts_model = initialize_xtts_model()
            return gr.update(value="✅ Modell erfolgreich geladen", visible=True)
        except Exception as e:
            return gr.update(value=f"❌ Modellfehler: {str(e)}", visible=True)
    
    with gr.Tab("🎙️ Stimmen Management"):
        # Modell-Status Label INNERHALB des Tabs
        model_status = gr.Label(
            value="🔄 Modell wird geladen...", 
            visible=True
        )

        with gr.Row():
            text_input = gr.Textbox(
                label="Standard-Text für Generierungen", 
                value=default_text,
                interactive=True
            )
        
        # GPT Conditioning Latents
        gr.Markdown("### GPT Conditioning Latents")
        with gr.Group():
            with gr.Row():
                # Erste GPT Conditioning Latent
                with gr.Column(scale=3):
                    gpt_latent1_dropdown = gr.Dropdown(
                        label="GPT Latent 1", 
                        choices=available_latents,
                        interactive=True
                    )
                    gpt_latent1_slider = gr.Slider(
                        minimum=0, 
                        maximum=100, 
                        value=50, 
                        label="Gewichtung Latent 1"
                    )
                    gpt_latent1_generate_btn = gr.Button("Generiere GPT Latent 1")
                    gpt_latent1_audio = gr.Audio(label="GPT Latent 1 Audio")
                
                # Zweite GPT Conditioning Latent
                with gr.Column(scale=3):
                    gpt_latent2_dropdown = gr.Dropdown(
                        label="GPT Latent 2", 
                        choices=available_latents,
                        interactive=True
                    )
                    gpt_latent2_slider = gr.Slider(
                        minimum=0, 
                        maximum=100, 
                        value=50, 
                        label="Gewichtung Latent 2"
                    )
                    gpt_latent2_generate_btn = gr.Button("Generiere GPT Latent 2")
                    gpt_latent2_audio = gr.Audio(label="GPT Latent 2 Audio")
        
        # Speaker Embedding Latents
        gr.Markdown("### Speaker Embedding Latents")
        with gr.Group():
            with gr.Row():
                # Erste Speaker Embedding Latent
                with gr.Column(scale=3):
                    speaker_latent1_dropdown = gr.Dropdown(
                        label="Speaker Latent 1", 
                        choices=available_latents,
                        interactive=True
                    )
                    speaker_latent1_slider = gr.Slider(
                        minimum=0, 
                        maximum=100, 
                        value=50, 
                        label="Gewichtung Latent 1"
                    )
                    speaker_latent1_generate_btn = gr.Button("Generiere Speaker Latent 1")
                    speaker_latent1_audio = gr.Audio(label="Speaker Latent 1 Audio")
                
                # Zweite Speaker Embedding Latent
                with gr.Column(scale=3):
                    speaker_latent2_dropdown = gr.Dropdown(
                        label="Speaker Latent 2", 
                        choices=available_latents,
                        interactive=True
                    )
                    speaker_latent2_slider = gr.Slider(
                        minimum=0, 
                        maximum=100, 
                        value=50, 
                        label="Gewichtung Latent 2"
                    )
                    speaker_latent2_generate_btn = gr.Button("Generiere Speaker Latent 2")
                    speaker_latent2_audio = gr.Audio(label="Speaker Latent 2 Audio")
        
        # Merge Bereich
        gr.Markdown("### Latent Merge")
        with gr.Group():
            with gr.Row():
                merge_name_input = gr.Textbox(label="Name des gemergten Latents")
                merge_btn = gr.Button("Merge Latents")
                merge_generate_btn = gr.Button("Generiere Merged Latent Audio")
                merged_audio = gr.Audio(label="Merged Latent Audio")
                
        # Slider-Synchronisation       
        def update_sliders(slider1: float, slider2: float, changed: str) -> tuple[gr.Slider, gr.Slider]:
            """
            Synchronisiert zwei Slider, sodass sie immer 100% ergeben.

            Args:
                slider1 (float): Wert des ersten Sliders.
                slider2 (float): Wert des zweiten Sliders.
                changed (str): Identifier, welcher Slider geändert wurde.

            Returns:
                tuple[gr.Slider, gr.Slider]: Aktualisierte Slider.
            """
            total = slider1 + slider2
            if total != 100:
                if changed == "gpt_latent1":
                    slider2 = 100 - slider1
                else:
                    slider1 = 100 - slider2
            return gr.Slider(value=slider1), gr.Slider(value=slider2)
        
        # Slider-Synchronisations-Event-Handler
        gpt_latent1_slider.change(
            fn=update_sliders, 
            inputs=[gpt_latent1_slider, gpt_latent2_slider, gr.State("gpt_latent1")], 
            outputs=[gpt_latent1_slider, gpt_latent2_slider]
        )
        gpt_latent2_slider.change(
            fn=update_sliders, 
            inputs=[gpt_latent1_slider, gpt_latent2_slider, gr.State("gpt_latent2")], 
            outputs=[gpt_latent1_slider, gpt_latent2_slider]
        )
        speaker_latent1_slider.change(
            fn=update_sliders, 
            inputs=[speaker_latent1_slider, speaker_latent2_slider, gr.State("speaker_latent1")], 
            outputs=[speaker_latent1_slider, speaker_latent2_slider]
        )
        speaker_latent2_slider.change(
            fn=update_sliders, 
            inputs=[speaker_latent1_slider, speaker_latent2_slider, gr.State("speaker_latent2")], 
            outputs=[speaker_latent1_slider, speaker_latent2_slider]
        )
        
        # Dropdown-Aktualisierung mit Fehlerbehandlung
        def update_latent_dropdowns():
            """Aktualisiert die Latent-Dropdowns mit verfügbaren Latent-Dateien"""
            try:
                # Suche nach JSON-Dateien in den Latent-Verzeichnissen
                latent_dirs = [
                    "/media/fukuro/raid5/alltalk_tts/voices/xtts_latents"
                ]
                
                # Sammle verfügbare Latents
                available_latents = []
                for directory in latent_dirs:
                    latent_files = glob.glob(os.path.join(directory, "*.json"))
                    available_latents.extend([os.path.splitext(os.path.basename(f))[0] for f in latent_files])
                
                # Debugging-Ausgabe
                print(f"🔍 Verfügbare Latents: {available_latents}")
                
                # Wenn keine Latents gefunden wurden, gib eine Standardoption
                if not available_latents:
                    available_latents = ["Keine Latents gefunden"]
                
                # Rückgabe der Dropdown-Updates
                return (
                    gr.update(choices=available_latents),  # gpt_latent1_dropdown
                    gr.update(choices=available_latents),  # gpt_latent2_dropdown
                    gr.update(choices=available_latents),  # speaker_latent1_dropdown
                    gr.update(choices=available_latents)   # speaker_latent2_dropdown
                )
            
            except Exception as e:
                # Umfangreiches Debugging für unerwartete Fehler
                import traceback
                print(f"❌ Fehler bei Dropdown-Aktualisierung: {e}")
                print("🔍 Traceback:")
                traceback.print_exc()
                
                # Fallback-Rückgabe im Fehlerfall
                return (
                    gr.update(choices=["Fehler"]),
                    gr.update(choices=["Fehler"]),
                    gr.update(choices=["Fehler"]),
                    gr.update(choices=["Fehler"])
                )
        
        load_btn = gr.Button(value="Latent Merge Model laden")
        progress_load = gr.Label(label="Progress:")
        
        load_btn.click(
            fn=update_latent_dropdowns,
            inputs=[],
            outputs=[gpt_latent1_dropdown, gpt_latent2_dropdown, speaker_latent1_dropdown, speaker_latent2_dropdown]
        )

        # Generierungs-Handler für Latents
        gpt_latent1_generate_btn.click(
            fn=generate_latent_audio,
            inputs=[gpt_latent1_dropdown, text_input],
            outputs=[gpt_latent1_audio]
        )
        gpt_latent2_generate_btn.click(
            fn=generate_latent_audio,
            inputs=[gpt_latent2_dropdown, text_input],
            outputs=[gpt_latent2_audio]
        )
        speaker_latent1_generate_btn.click(
            fn=generate_latent_audio,
            inputs=[speaker_latent1_dropdown, text_input],
            outputs=[speaker_latent1_audio]
        )
        speaker_latent2_generate_btn.click(
            fn=generate_latent_audio,
            inputs=[speaker_latent2_dropdown, text_input],
            outputs=[speaker_latent2_audio]
        )
        
        # Merge-Handler
        def merge_latents_for_ui(gpt1, gpt1_weight, gpt2, gpt2_weight, 
                          speaker1, speaker1_weight, speaker2, speaker2_weight,
                          merge_name):
            """
            Merge verschiedene Latents für die Benutzeroberfläche (Wrapper für die allgemeinere merge_latents-Funktion)
            
            Args:
                gpt1 (str): Erster GPT-Latent
                gpt1_weight (float): Gewichtung für ersten GPT-Latent
                gpt2 (str): Zweiter GPT-Latent
                gpt2_weight (float): Gewichtung für zweiten GPT-Latent
                speaker1 (str): Erster Speaker-Latent
                speaker1_weight (float): Gewichtung für ersten Speaker-Latent
                speaker2 (str): Zweiter Speaker-Latent
                speaker2_weight (float): Gewichtung für zweiten Speaker-Latent
                merge_name (str): Name für den gemergten Latent
            
            Returns:
                str: Dateiname des gemergten Latents
            """
            try:
                # Lade Latent-Dateien
                latent_dir = get_latent_directory()
                
                # Vorbereitung der Latent-Quellen
                latent_sources = {
                    'gpt_cond_latent': [
                        {'json': json.load(open(os.path.join(latent_dir, f"{gpt1}.json"))), 'weight': gpt1_weight/100, 'name': gpt1},
                        {'json': json.load(open(os.path.join(latent_dir, f"{gpt2}.json"))), 'weight': gpt2_weight/100, 'name': gpt2}
                    ],
                    'speaker_embedding': [
                        {'json': json.load(open(os.path.join(latent_dir, f"{speaker1}.json"))), 'weight': speaker1_weight/100, 'name': speaker1},
                        {'json': json.load(open(os.path.join(latent_dir, f"{speaker2}.json"))), 'weight': speaker2_weight/100, 'name': speaker2}
                    ]
                }
                
                # Merge-Vorgang
                merged_latent = merge_latents(latent_sources)
                
                # Generiere Dateinamen
                merged_filename = generate_merge_filename(latent_sources)
                
                # Speichere gemergten Latent
                merged_path = os.path.join(latent_dir, merged_filename)
                with open(merged_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_latent, f, indent=2)
                
                return merged_filename
            
            except Exception as e:
                print(f"❌ Fehler beim Mergen der Latents: {e}")
                return None
        
        merge_btn.click(
            fn=merge_latents_for_ui,
            inputs=[
                gpt_latent1_dropdown, gpt_latent1_slider, 
                gpt_latent2_dropdown, gpt_latent2_slider,
                speaker_latent1_dropdown, speaker_latent1_slider,
                speaker_latent2_dropdown, speaker_latent2_slider,
                merge_name_input
            ],
            outputs=[merge_name_input]
        )
        
        merge_generate_btn.click(
            fn=lambda name, text: generate_latent_audio(prepare_latent_name(name), text),
            inputs=[merge_name_input, text_input],
            outputs=[merged_audio]
        )
        
        # Latent-Ähnlichkeitsberechnung
        gpt_similarity_btn, gpt_similarity_output, speaker_similarity_btn, speaker_similarity_output = setup_similarity_buttons(
            demo, 
            gpt_latent1_dropdown, 
            gpt_latent2_dropdown,
            speaker_latent1_dropdown, 
            speaker_latent2_dropdown
        )
        
        # Füge Ähnlichkeitsberechnung hinzu
        gr.Markdown("### Latent Ähnlichkeitsberechnung")
        with gr.Group():
            with gr.Row():
                gpt_similarity_btn
                gpt_similarity_output
            with gr.Row():
                speaker_similarity_btn
                speaker_similarity_output
    
    # Hintergrund-Initialisierung starten
    demo.load(
        fn=background_model_init, 
        inputs=None, 
        outputs=[model_status]
    )

    return demo

def initialize_xtts_model():
    """
    Initialisiert das XTTS-Modell für Latent-Generierung.
    
    Returns:
        Xtts: Initialisiertes und geladenes XTTS-Modell
    """
    config_path = "/media/fukuro/raid5/alltalk_tts/models/xtts/xttsv2_2.0.3/config.json"
    checkpoint_path = "/media/fukuro/raid5/alltalk_tts/models/xtts/xttsv2_2.0.3/model.pth"
    vocab_path = "/media/fukuro/raid5/alltalk_tts/models/xtts/xttsv2_2.0.3/vocab.json"
    
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        use_deepspeed=False
    )
    if torch.cuda.is_available():
        model.cuda()
    
    return model

import torch
import numpy as np

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

def setup_similarity_buttons(
    demo: gr.Blocks, 
    gpt_latent1_dropdown: gr.Dropdown, 
    gpt_latent2_dropdown: gr.Dropdown,
    speaker_latent1_dropdown: gr.Dropdown, 
    speaker_latent2_dropdown: gr.Dropdown
) -> tuple:
    """
    Richtet Buttons und Ausgabefelder für Latent-Ähnlichkeitsberechnung ein.
    
    Args:
        demo (gr.Blocks): Gradio Blocks-Instanz
        gpt_latent1_dropdown (gr.Dropdown): Dropdown für ersten GPT Latent
        gpt_latent2_dropdown (gr.Dropdown): Dropdown für zweiten GPT Latent
        speaker_latent1_dropdown (gr.Dropdown): Dropdown für ersten Speaker Latent
        speaker_latent2_dropdown (gr.Dropdown): Dropdown für zweiten Speaker Latent
    
    Returns:
        tuple: Buttons und Ausgabefelder für Ähnlichkeitsberechnung
    """
    def calculate_gpt_latent_similarity(gpt_latent1, gpt_latent2):
        """Berechnet Ähnlichkeit zwischen zwei GPT Latents"""
        try:
            latent1 = load_latent_from_json(gpt_latent1)
            latent2 = load_latent_from_json(gpt_latent2)
            similarity = calculate_latent_similarity(latent1, latent2)
            return f"Gesamtähnlichkeit: {similarity['total_similarity']:.6f} %\nGPT Ähnlichkeit: {similarity['gpt_similarity']:.6f} %\nSpeaker Ähnlichkeit: {similarity['speaker_similarity']:.6f} %"
        except Exception as e:
            return f"Fehler: {str(e)}"

    def calculate_speaker_latent_similarity(speaker_latent1, speaker_latent2):
        """Berechnet Ähnlichkeit zwischen zwei Speaker Latents"""
        try:
            latent1 = load_latent_from_json(speaker_latent1)
            latent2 = load_latent_from_json(speaker_latent2)
            similarity = calculate_latent_similarity(latent1, latent2)
            return f"Gesamtähnlichkeit: {similarity['total_similarity']:.6f} %\nGPT Ähnlichkeit: {similarity['gpt_similarity']:.6f} %\nSpeaker Ähnlichkeit: {similarity['speaker_similarity']:.6f} %"
        except Exception as e:
            return f"Fehler: {str(e)}"

    # GPT Latent Ähnlichkeitsberechnung
    gpt_similarity_btn = gr.Button("🔍 GPT Latents vergleichen")
    gpt_similarity_output = gr.Textbox(label="GPT Latent Ähnlichkeit")
    
    gpt_similarity_btn.click(
        fn=calculate_gpt_latent_similarity,
        inputs=[gpt_latent1_dropdown, gpt_latent2_dropdown],
        outputs=gpt_similarity_output
    )

    # Speaker Latent Ähnlichkeitsberechnung
    speaker_similarity_btn = gr.Button("🔍 Speaker Latents vergleichen")
    speaker_similarity_output = gr.Textbox(label="Speaker Latent Ähnlichkeit")
    
    speaker_similarity_btn.click(
        fn=calculate_speaker_latent_similarity,
        inputs=[speaker_latent1_dropdown, speaker_latent2_dropdown],
        outputs=speaker_similarity_output
    )

    return (
        gpt_similarity_btn, 
        gpt_similarity_output, 
        speaker_similarity_btn, 
        speaker_similarity_output
    )
