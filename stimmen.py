"""
AllTalk TTS - Stimmen Management Modul

Dieses Modul bietet Funktionen für:
- Latent-Verzeichnis-Management
- Modell-Pfadkonstruktion
- Latent-Merge-Prozesse
- Audio-Generierung

Hauptfunktionen:
1. Verzeichnis-Funktionen:
   - get_latent_directory(): Ermittelt Latent-Verzeichnis
   - get_model_directory(): Bestimmt Modell-Verzeichnis
   - get_model_files(): Findet Modell-Dateien
   - find_best_models(): Sucht nach besten Modell-Dateien
   - find_jsons(): Findet spezifische JSON-Dateien

2. Latent-Management:
   - load_available_latents(): Lädt verfügbare Latent-JSONs
   - check_latent_compatibility(): Prüft Latent-Kompatibilität
   - merge_latents(): Merged Latents mit Gewichtung
   - generate_merge_filename(): Generiert Dateinamen für Merge

3. Audio-Generierung:
   - generate_latent_audio(): Generiert Audio aus Latent
   - setup_stimmen_tab(): Erstellt Benutzeroberfläche für Stimmen-Management

Hauptanwendungsbereich: Text-to-Speech Latent-Verarbeitung

Funktionen in Reihenfolge:

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

4. find_best_models():
   - Sucht nach 'best_model.pth' Dateien
   - Optional: Filterung nach Sprecher-Namen

5. find_jsons():
   - Findet spezifische JSON-Dateien
   - Optionale Filterung nach Sprecher-Namen

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

10. setup_stimmen_tab():
    - Erstellt komplexe Gradio-Benutzeroberfläche
    - Integriert Latent-Management-Funktionen
    - Implementiert Slider-Synchronisation
    - Handhabt Modell-Laden und Audio-Generierung

11. generate_latent_audio():
    - Generiert Audio aus Latent
    - Unterstützt benutzerdefinierte Texte
    - Fehlerbehandlung und Debugging
    - Temporäre Datei-Generierung
"""

import os
import json
import numpy as np
import gradio as gr
import glob
import logging
import time
from functools import wraps

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
                [
                    [np.average(
                        [source[j][0] for source in latent_data], 
                        weights=[src['normalized_weight'] for src in normalized_sources],
                        axis=0
                    )]
                ] 
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
    output_dir: str | None = None
) -> str | None:
    """
    Generiert Audio aus einem beliebigen Latent.

    Args:
        latent_name (str): Name des Latents (ohne .json-Erweiterung).
        text (str, optional): Text zur Audio-Generierung. 
            Verwendet Standardtext, wenn None. Defaults to None.
        output_dir (str, optional): Verzeichnis für generierte Audios. 
            Erstellt temporäres Verzeichnis, wenn None. Defaults to None.

    Returns:
        str | None: Pfad zur generierten Audiodatei oder None bei Fehler.

    Raises:
        ImportError: Wenn TTS-Engine nicht geladen werden kann.
        RuntimeError: Bei Problemen während der Audio-Generierung.
    """
    try:
        # Importiere die TTS-Engine
        from system.tts_engines.xtts.model_engine import tts_class
        import tempfile
        
        # Standardtext für deutsche Generierung
        if text is None:
            text = "Der alte Meister saß in seinem Studierzimmer und betrachtete die alten Schriftrollen."
        
        # Temporäres Verzeichnis für Vorschau-Audios
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix='alltalk_preview_')
        
        # Generiere Ausgabedateinamen im temporären Verzeichnis
        output_path = os.path.join(output_dir, f"{latent_name}_preview.wav")
        
        # Initialisiere TTS-Engine
        tts_engine = tts_class()
        
        # Audio generieren - WICHTIG: Prefix 'latent:' beibehalten!
        result = tts_engine.generate_tts(
            text=text,
            voice=f"latent:{latent_name}.json",  # Wichtig: Prefix 'latent:' bleibt!
            language="de",
            temperature=0.7,
            repetition_penalty=5.0,
            speed=1.0,
            pitch=0.0,
            output_file=output_path,
            streaming=False
        )
        
        # Überprüfe, ob Audio generiert wurde
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"❌ Keine Audio-Datei generiert: {output_path}")
            return None
        
        print(f"✅ Audio-Vorschau erfolgreich generiert: {output_path}")
        return output_path
    
    except Exception as e:
        # Umfangreiches Debugging
        import traceback
        print(f"❌ Fehler bei Audio-Generierung aus Latent: {e}")
        print("🔍 Traceback:")
        traceback.print_exc()
        return None

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
    
    with gr.Tab("🎙️ Stimmen Management"):
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
        
        # Modell-Laden und Dropdowns aktualisieren
        def load_model_for_merge():
            """Lädt das TTS-Modell für Latent Merge"""
            try:
                import torch
                from TTS.tts.models.xtts import Xtts
                from TTS.tts.configs.xtts_config import XttsConfig
                
                # Debugging: Überprüfe CUDA-Verfügbarkeit
                print(f"🔍 CUDA verfügbar: {torch.cuda.is_available()}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"🖥️ Verwendetes Gerät: {device}")
                
                # Modellpfad dynamisch ermitteln
                model_path = get_model_directory()
                
                # Modelldateien dynamisch laden
                model_files = get_model_files(model_path)
                
                # Latent-Verzeichnisse erstellen
                latent_dirs = [get_latent_directory()]
                for directory in latent_dirs:
                    os.makedirs(directory, exist_ok=True)
                    print(f"📂 Verzeichnis erstellt: {directory}")
                
                # Konfiguration laden
                config = XttsConfig()
                config.load_json(model_files['config'])
                
                # Modell initialisieren
                model = Xtts.init_from_config(config)
                
                # Modell laden
                model.load_checkpoint(
                    config,
                    checkpoint_path=model_files['checkpoint'],
                    vocab_path=model_files['vocab'],
                    use_deepspeed=False,
                    speaker_file_path=model_files['speakers']
                )
                
                # Auf GPU verschieben, falls verfügbar
                if torch.cuda.is_available():
                    model.cuda()
                
                print("✅ Modell erfolgreich geladen!")
                
                # Zusätzliche Überprüfung der Latent-Dateien
                latent_files = glob.glob(f"{latent_dirs[0]}/*")
                print(f"📋 Gefundene Latent-Dateien: {latent_files}")
                
                return "✅ Modell erfolgreich geladen! Latent Merge ist jetzt verfügbar."
            
            except ImportError as e:
                print(f"❌ Import-Fehler: {e}")
                return f"Import-Fehler: Konnte Modell nicht laden - {e}"
            
            except Exception as e:
                # Umfangreiches Debugging für unerwartete Fehler
                import traceback
                print(f"❌ Unerwarteter Fehler: {e}")
                print("🔍 Traceback:")
                traceback.print_exc()
                return f"Fehler beim Laden des Modells: {str(e)}"

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
            fn=load_model_for_merge,
            inputs=[],
            outputs=[progress_load]
        )
        load_btn.click(
            fn=update_latent_dropdowns,
            inputs=[],
            outputs=[
                gpt_latent1_dropdown, 
                gpt_latent2_dropdown, 
                speaker_latent1_dropdown, 
                speaker_latent2_dropdown
            ]
        )

        # Generierungs-Handler für Latents
        def generate_latent_audio(latent_name, text=None, output_dir=None):
            """
            Generiere Audio aus einem beliebigen Latent
            
            Args:
                latent_name (str): Name des Latents (ohne .json)
                text (str, optional): Text zur Audio-Generierung. 
                                       Verwendet Standardtext, wenn None
                output_dir (str, optional): Verzeichnis für generierte Audios
            
            Returns:
                str: Pfad zur generierten Audiodatei oder None bei Fehler
            """
            try:
                # Importiere die TTS-Engine
                from system.tts_engines.xtts.model_engine import tts_class
                import tempfile
                
                # Standardtext für deutsche Generierung
                if text is None:
                    text = "Der alte Meister saß in seinem Studierzimmer und betrachtete die alten Schriftrollen."
                
                # Temporäres Verzeichnis für Vorschau-Audios
                if output_dir is None:
                    output_dir = tempfile.mkdtemp(prefix='alltalk_preview_')
                
                # Generiere Ausgabedateinamen im temporären Verzeichnis
                output_path = os.path.join(output_dir, f"{latent_name}_preview.wav")
                
                # Initialisiere TTS-Engine
                tts_engine = tts_class()
                
                # Audio generieren - WICHTIG: Prefix 'latent:' beibehalten!
                result = tts_engine.generate_tts(
                    text=text,
                    voice=f"latent:{latent_name}.json",  # Wichtig: Prefix 'latent:' bleibt!
                    language="de",
                    temperature=0.7,
                    repetition_penalty=5.0,
                    speed=1.0,
                    pitch=0.0,
                    output_file=output_path,
                    streaming=False
                )
                
                # Überprüfe, ob Audio generiert wurde
                if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    print(f"❌ Keine Audio-Datei generiert: {output_path}")
                    return None
                
                print(f"✅ Audio-Vorschau erfolgreich generiert: {output_path}")
                return output_path
            
            except Exception as e:
                # Umfangreiches Debugging
                import traceback
                print(f"❌ Fehler bei Audio-Generierung aus Latent: {e}")
                print("🔍 Traceback:")
                traceback.print_exc()
                return None
        
        # Generierungs-Event-Handler für Latents
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
            Merge verschiedene Latents für die Benutzeroberfläche
            
            Args:
                gpt1, gpt2: Namen der GPT-Latents
                gpt1_weight, gpt2_weight: Gewichtungen für GPT-Latents
                speaker1, speaker2: Namen der Speaker-Latents
                speaker1_weight, speaker2_weight: Gewichtungen für Speaker-Latents
                merge_name: Name für den gemergten Latent
            
            Returns:
                str: Dateiname des gemergten Latents
            """
            try:
                # Hole Latent-Verzeichnis
                latent_dir = get_latent_directory()
                
                # Lade verfügbare Latents zur Überprüfung
                available_latents = load_available_latents()
                print(f"📋 Verfügbare Latents: {available_latents}")
                
                # Lade JSONs
                gpt_sources = [
                    {'json': json.load(open(os.path.join(latent_dir, f"{gpt1}.json"))), 'weight': gpt1_weight/100, 'name': gpt1},
                    {'json': json.load(open(os.path.join(latent_dir, f"{gpt2}.json"))), 'weight': gpt2_weight/100, 'name': gpt2}
                ]
                speaker_sources = [
                    {'json': json.load(open(os.path.join(latent_dir, f"{speaker1}.json"))), 'weight': speaker1_weight/100, 'name': speaker1},
                    {'json': json.load(open(os.path.join(latent_dir, f"{speaker2}.json"))), 'weight': speaker2_weight/100, 'name': speaker2}
                ]
                
                # Prüfe Kompatibilität
                check_latent_compatibility([s['json'] for s in gpt_sources + speaker_sources])
                
                # Merge Latents
                merged_latents = merge_latents({
                    'gpt_cond_latent': gpt_sources,
                    'speaker_embedding': speaker_sources
                })
                
                # Generiere Dateinamen
                merge_filename = generate_merge_filename({
                    'gpt_cond_latent': gpt_sources,
                    'speaker_embedding': speaker_sources
                })
                
                # Speichere gemergten Latent
                merged_path = os.path.join(latent_dir, merge_filename)
                with open(merged_path, 'w') as f:
                    json.dump(merged_latents, f, indent=2)
                
                return merge_filename
            except Exception as e:
                import traceback
                print(f"❌ Fehler beim Mergen der Latents: {e}")
                traceback.print_exc()
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
            fn=generate_latent_audio,
            inputs=[merge_name_input, text_input],
            outputs=[merged_audio]
        )
    
    return demo
