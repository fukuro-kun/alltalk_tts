import os
import json
import torch
import torchaudio
import tempfile
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def generate_latent_preview(latent_path, text="Der alte Meister saß in seinem Studierzimmer und betrachtete die alten Schriftrollen."):
    """Generiere eine Audio-Vorschau aus einer Latent-Datei"""
    # Latent-Datei laden
    with open(latent_path, 'r') as f:
        latent_data = json.load(f)
    
    # Lade das XTTS-Modell (einmalig)
    if not hasattr(generate_latent_preview, 'model'):
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
        
        generate_latent_preview.model = model
    
    # Generiere Audio
    gpt_cond_latent = torch.tensor(latent_data['gpt_cond_latent'])
    speaker_embedding = torch.tensor(latent_data['speaker_embedding'])
    
    out = generate_latent_preview.model.inference(
        text=text,
        language="de",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7,
        length_penalty=1.0,
        repetition_penalty=2.0
    )
    
    # Speichere Audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(str(out_path), out["wav"], 24000)
    
    return out_path

def generate_all_latent_previews(latent_dir="/media/fukuro/raid5/alltalk_tts/voices/xtts_latents"):
    """Generiere Vorschauen für alle Latent-Dateien"""
    latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.json')]
    
    previews = {}
    for latent_file in latent_files:
        latent_path = os.path.join(latent_dir, latent_file)
        preview_path = generate_latent_preview(latent_path)
        previews[latent_file] = preview_path
    
    return previews

if __name__ == "__main__":
    preview_paths = generate_all_latent_previews()
    for latent_file, preview_path in preview_paths.items():
        print(f"Vorschau für {latent_file}: {preview_path}")
