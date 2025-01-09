#!/usr/bin/env python3
"""
Einfacher Test-Client für die AllTalk TTS-API.

Verwendung:
    python test_client.py [--text {kurz,mittel,lang}] [--url URL] [--output DIR] [--speaker NAME]

Argumente:
    --text      Wählt den Test-Text aus (Standard: "mittel")
                - kurz:   Ein kurzer Testsatz (~10 Wörter)
                - mittel: Ein mittellanger Paragraph (~50 Wörter)
                - lang:   Ein langer Text mit mehreren Absätzen (~250 Wörter)
    
    --url       Die URL des AllTalk-API Servers (Standard: http://localhost:7851)
    
    --output    Ausgabeverzeichnis für die generierten Audiodateien
                (Standard: ./output)

    --speaker   Name des zu verwendenden Sprechers
                (Standard: "Tom5.wav")
"""

import requests
import json
import argparse
import time
from pathlib import Path
from typing import Optional
import soundfile as sf
import os

# Test-Texte
SAMPLE_TEXTS = {
    "kurz": "Dies ist ein kurzer Test-Text für die TTS-API.",
    "mittel": "Der alte Meister saß in seinem Studierzimmer und betrachtete die alten Schriftrollen. Die Zeit schien stillzustehen, während er die vergilbten Seiten durchblätterte! Draußen tobte ein Sturm, aber hier drinnen war es warm und gemütlich.",
    "halblang":  """Hallo, mein Name ist Fred Meier und ich liebe die Farbe pink. Mein Lieblingsfach ist natürlich Englisch und zu Weihnachten wünsche ich mir ein Einhorn. Am liebsten mit Flügeln und in rosa und weiß. Damit möchte ich dann die ganzen Ferien über spielen. Was sollte man noch über mich wissen? Wenn ich nicht Lehrer geworden wäre, wäre ich Rockstar geworden.""",
    "lang": """Der alte Meister saß in seinem Studierzimmer und betrachtete die alten Schriftrollen. Die Zeit schien stillzustehen, während er die vergilbten Seiten durchblätterte! Draußen tobte ein Sturm, aber hier drinnen war es warm und gemütlich. Die Kerzen warfen flackernde Schatten an die Wände; manchmal tanzten sie wie kleine Geister über die Bücherregale.
    Der Meister nahm einen Schluck von seinem Tee (der schon lange kalt geworden war) und seufzte zufrieden. Die alten Texte faszinierten ihn immer wieder aufs Neue. Jede Seite enthielt Geheimnisse, die nur darauf warteten, entdeckt zu werden! In den verstaubten Regalen seiner Bibliothek lagerten Schätze des Wissens, die seit Jahrhunderten unberührt geblieben waren.
    "Interessant", murmelte er, als er eine besonders alte Passage entdeckte. Die Schrift war verblasst, aber noch lesbar; der Text handelte von einem uralten Ritual. Vorsichtig legte er das Pergament beiseite und griff nach seinem Notizbuch. Die Übersetzung werde Zeit brauchen, aber wie viel genau? Aber das machte ihm nichts aus.
    Die Nacht zog herauf, und der Sturm draußen wurde stärker. Regen peitschte gegen die Fenster, und der Wind heulte um die alten Mauern! Doch der Meister bemerkte nichts davon; er war völlig in seine Arbeit vertieft. Seine Feder kratzte über das Papier, während er sorgsam jedes Detail notierte.
    Stunde um Stunde verging, bis der Morgen dämmerte. Die ersten Sonnenstrahlen fielen durch die hohen Fenster und tauchten den Raum in goldenes Licht; die Schatten der Nacht wichen wieder. Der Meister rieb sich die müden Augen und betrachtete zufrieden seine Notizen. Es war eine produktive Nacht gewesen."""
}

def get_audio_duration(audio_path: str) -> float:
    """
    Ermittelt die Dauer einer Audiodatei in Sekunden.
    
    Args:
        audio_path (str): Pfad zur Audiodatei
    
    Returns:
        float: Länge der Audiodatei in Sekunden
    """
    try:
        with sf.SoundFile(audio_path) as f:
            return len(f) / f.samplerate
    except Exception as e:
        print(f"Fehler beim Ermitteln der Audiodauer: {e}")
        return 0.0

def calculate_performance_metrics(text: str, audio_duration: float, 
                                 generation_time: float) -> dict:
    """
    Berechnet detaillierte Performance-Metriken für TTS-Generierung.
    
    Args:
        text (str): Eingabetext
        audio_duration (float): Länge des generierten Audios in Sekunden
        generation_time (float): Gesamte Generierungszeit in Sekunden
    
    Returns:
        dict: Detaillierte Performance-Metriken
    """
    # Wörter und Zeichen zählen
    words = len(text.split())
    chars = len(text)

    # Performance-Metriken berechnen
    metrics = {
        'words_total': words,
        'chars_total': chars,
        'words_per_generation_second': round(words / generation_time, 2) if generation_time > 0 else 0,
        'chars_per_generation_second': round(chars / generation_time, 2) if generation_time > 0 else 0,
        'audio_duration': round(audio_duration, 2),
        'generation_time': round(generation_time, 3),
        'speed_factor': round(audio_duration / generation_time, 3) if generation_time > 0 else 0
    }

    return metrics

def generate_tts(text: str, url: str = "http://localhost:7851", 
                 output_dir: Optional[str] = None, 
                 speaker: str = "Tom5.wav") -> Path:
    """
    Sendet eine TTS-Anfrage an den AllTalk-Server.
    
    Args:
        text (str): Der zu generierende Text
        url (str, optional): Server-URL. Defaults to "http://localhost:7851".
        output_dir (Optional[str], optional): Ausgabeverzeichnis. Defaults to None.
        speaker (str, optional): Name des Sprechers. Defaults to "Tom5.wav".
    
    Returns:
        Path: Pfad zur generierten Audiodatei
    """
    # Ausgabeverzeichnis vorbereiten
    output_dir = Path(output_dir or "/media/fukuro/raid5/alltalk_tts/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zeitstempel für eindeutigen Dateinamen
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"tts_output_{timestamp}.wav"

    # Gesamte Generierungszeit messen
    total_start_time = time.time()

    # Payload für TTS-Anfrage
    payload = {
        "text_input": text,
        "character_voice_gen": speaker,
        "output_file_name": output_file.stem,
        "language": "de"  # Standardsprache
    }

    try:
        # Zeitpunkt vor der Anfrage
        request_start_time = time.time()
        
        response = requests.post(f"{url}/api/tts-generate", data=payload)
        response.raise_for_status()

        # Zeitpunkt nach der Antwort
        request_end_time = time.time()
        generation_time = request_end_time - request_start_time

        # Antwort verarbeiten
        result = response.json()

        # TTS-Anfrage Details
        print(f"📋 TTS-Anfrage am {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔍 Anfrage-Details:")
        
        # Text-Vorschau (max. 80 Zeichen)
        text_preview = (text[:80] + '...') if len(text) > 80 else text
        print(f"   text: {text_preview}")

        # Optional: Formatierte Server-Antwort
        print("\n🌐 Detaillierte Server-Antwort:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Suche nach der tatsächlichen Ausgabedatei
        output_files = list(output_dir.glob(f"{output_file.stem}*"))
        if output_files:
            output_file = output_files[0]

        # Audio-Dauer ermitteln
        audio_duration = get_audio_duration(str(output_file))

        # Performance-Metriken berechnen
        metrics = calculate_performance_metrics(
            text, 
            audio_duration, 
            generation_time
        )

        # Performance-Metriken ausgeben
        print(f"\n📂 Server-Rückgabe am {time.strftime('%Y-%m-%d %H:%M:%S')}:")
        print(f"   🗃️  Ursprünglicher Name:  {output_file.name}")
        print(f"   📂 Ursprünglicher Pfad:  {output_file}")

        print(f"\n✅ Audio generiert: {output_file}")
        print(f"📊 Performance-Metriken:")
        print(f"   🔤 Zeichen gesamt:     {metrics['chars_total']}")
        print(f"   📝 Wörter gesamt:      {metrics['words_total']}")
        print(f"   🚀 Wörter/Generierungssekunde: {metrics['words_per_generation_second']}")
        print(f"   📜 Zeichen/Generierungssekunde: {metrics['chars_per_generation_second']}")
        print(f"🕰️  Audio-Dauer:         {metrics['audio_duration']} Sekunden")
        print(f"🔄 Generierungszeit:     {metrics['generation_time']} Sekunden")
        print(f"🏎️  Geschwindigkeitsfaktor: {metrics['speed_factor']}x")

        return output_file

    except requests.exceptions.RequestException as e:
        print(f"❌ Fehler bei der TTS-Generierung: {e}")
        raise

def main():
    """
    Hauptfunktion zum Ausführen des Test-Clients.
    """
    parser = argparse.ArgumentParser(description="AllTalk TTS Test-Client")
    parser.add_argument("--text", choices=["kurz", "mittel", "halblang", "lang"], 
                        default="mittel", help="Länge des Test-Texts")
    parser.add_argument("--url", default="http://localhost:7851", 
                        help="URL des AllTalk TTS Servers")
    parser.add_argument("--output", help="Ausgabeverzeichnis für Audiodateien")
    parser.add_argument("--speaker", default="Tom5.wav", 
                        help="Name des Sprechers")

    args = parser.parse_args()

    # Text auswählen
    text = SAMPLE_TEXTS[args.text]

    # TTS generieren
    generate_tts(
        text=text, 
        url=args.url, 
        output_dir=args.output, 
        speaker=args.speaker
    )

if __name__ == "__main__":
    main()
