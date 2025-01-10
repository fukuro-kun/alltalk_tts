# Refactoring Plan: Latent Audio Preview Integration

## Kernziele 🎯
- Modellladelogik aus generate_latent_previews.py übernehmen
- Audio-Generierung vereinfachen
- Minimale Änderungen an bestehender Architektur

🎯 Übergeordnetes Ziel
Minimale Änderungen mit Fokus auf:

Modellladen verbessern
Audio-Generierung reparieren
KISS-Prinzip befolgen
🔍 Detaillierte Funktions-Analyse
1-3. Basis-Utility-Funktionen (KEINE ÄNDERUNGEN)
get_latent_directory()
get_model_directory()
get_model_files() ✅ Funktionen sind gut, keine Modifikation nötig
4-5. Such- und Filterfunktionen (KEINE ÄNDERUNGEN)
find_best_models()
find_jsons() ✅ Funktionen sind gut, keine Modifikation nötig
6. load_available_latents() (KEINE ÄNDERUNGEN)
✅ Funktion ist gut, keine Modifikation nötig

7. check_latent_compatibility() (KEINE ÄNDERUNGEN)
✅ Funktion ist gut, keine Modifikation nötig

8. merge_latents() (KEINE ÄNDERUNGEN)
✅ Funktion ist gut, keine Modifikation nötig

9. generate_merge_filename() (KEINE ÄNDERUNGEN)
✅ Funktion ist gut, keine Modifikation nötig

10. generate_latent_audio() (HAUPTFOKUS)
🔧 Zu ändern:

Komplette Neuimplementierung
Logik aus generate_latent_previews.py übernehmen
Vereinfachte Fehlerbehandlung
Statische Modell-Initialisierung

### Implementierungsansatz
```python
def generate_latent_audio(
    latent_name: str, 
    text: str | None = None, 
    output_dir: str | None = None,
    model_name: str = "xtts",
    version: str = "2.0.3"
):
    # Statische Modell-Initialisierung
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
```



## Modell-Initialisierungsstrategie

### Anforderungen
- Dropdowns ZUERST befüllen
- Modell-Initialisierung parallel starten
- Keine Blockierung der Benutzeroberfläche
- Fortschritt für Benutzer sichtbar

11. setup_stimmen_tab() (MINIMALE ÄNDERUNGEN)
🔧 Zu ändern:

Dropdown-Initialisierung vorziehen
Modell-Initialisierung im Hintergrund
Progress-Tracking hinzufügen

### Implementierungsansatz
```python
def setup_stimmen_tab(demo: gr.Blocks) -> gr.Blocks:
    # 1. Dropdowns schnell befüllen
    available_latents = load_available_latents()
    
    # Dropdown-Komponenten vorbereiten
    gpt_latent1_dropdown = gr.Dropdown(
        label="GPT Latent 1", 
        choices=available_latents
    )
    # ... weitere Dropdowns analog

    # 2. Hintergrund-Modell-Initialisierung
    def background_model_init():
        config = XttsConfig()
        config.load_json(config_path)
        xtts_model = Xtts.init_from_config(config)
        xtts_model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            use_deepspeed=False
        )
        if torch.cuda.is_available():
            xtts_model.cuda()
        
        # Modell als Attribut des Tabs speichern
        demo.xtts_model = xtts_model
        return "Modell geladen ✅"

    # 3. Progress-Tracking
    model_load_status = gr.Label(value="Modell wird geladen...")
    gr.Button("Modell laden").click(
        fn=background_model_init, 
        outputs=[model_load_status]
    )
```


12. initialize_xtts_model() (NEU)
🆕 Neue Funktion:

### Implementierungsansatz
```python
def initialize_xtts_model():
    """
    Initialisiert das XTTS-Modell für Latent-Generierung.
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
 ```   
🚀 Zusammenfassung der Änderungen
Nur generate_latent_audio() und setup_stimmen_tab() signifikant modifiziert
Neue initialize_xtts_model() Funktion
Alle anderen Funktionen unverändert
Fokus auf KISS-Prinzip
Modellladen und Audio-Generierung verbessert

## Vorteile
- Sofortige Dropdown-Befüllung
- Parallele Modell-Initialisierung
- Keine Blockierung der UI
- Benutzer-Feedback

## Zu beachten
- Temporäre Dateien sauber behandeln
- Konsistenz mit bestehender Architektur

## Nächste Schritte
- [x] Implementierung in stimmen.py
  - [x] `initialize_xtts_model()` überprüfen und übernehmen
  - [x] `generate_latent_audio()` komplett neu implementieren
  - [x] `setup_stimmen_tab()` anpassen
    - [x] Dropdowns zuerst befüllen
    - [x] Modell-Initialisierung im Hintergrund
    - [x] Progress-Tracking hinzufügen
- [ ] Kurzer Test der Funktionalität
- [ ] Integration in Gradio-Oberfläche
