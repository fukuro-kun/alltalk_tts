import os
import re
import pandas as pd
import gradio as gr

def setup_stimmen_tab(demo):
    """
    Tab für Stimmenverwaltung mit Ranking-Funktionalität
    
    Args:
        demo (gr.Blocks): Hauptdemonstrations-Objekt
    """
    speaker_dir = '/home/fukuro/Musik/sprecher/fertig/'
    
    with gr.Tab("🎙️ Stimmen Management"):
        gr.Markdown("# 🎙️ Stimmen Management")
        
        # Zustand für die aktuell ausgewählte Zeile
        selected_row_state = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=3):  # Schmaler Spaltenbereich für Tabelle
                speaker_table = gr.Dataframe(
                    headers=['Audio', 'Vorname', 'Name', 'Ranking'],  # "Play" entfernt
                    datatype=['str', 'str', 'str', 'number'],
                    type='pandas',
                    height=350,  # Fixe Höhe von 350 Pixeln
                    elem_id="speaker-table"  # Eindeutige ID für JavaScript-Manipulation
                )
            
            with gr.Column(scale=1):  # Kleinere Spalte für Audio und Buttons
                # Kompakte Audio-Vorschau mit Edit-Modus
                audio_preview = gr.Audio(
                    type="filepath", 
                    label="Audio Vorschau",
                    interactive=True,
                    autoplay=True,
                    show_download_button=False,
                    editable=True  # Edit-Modus aktivieren
                )
                
                # Buttons in derselben Spalte
                with gr.Row():
                    move_up_btn = gr.Button("⬆️ Ausgewählten Eintrag nach oben")
                    move_down_btn = gr.Button("⬇️ Ausgewählten Eintrag nach unten")
                
                with gr.Row():
                    refresh_btn = gr.Button("Sprecher aktualisieren")
                    save_ranking_btn = gr.Button("Ranking speichern")
        
        # JavaScript zum Fixieren der Tabellenhöhe
        demo.load(
            None, 
            None, 
            js="""
            function fixTableHeight() {
                const table = document.getElementById('speaker-table');
                if (table) {
                    const tableWrapper = table.closest('.table-wrap');
                    if (tableWrapper) {
                        tableWrapper.style.height = '350px';
                        tableWrapper.style.maxHeight = '350px';
                        tableWrapper.style.overflowY = 'auto';
                        
                        // Zusätzliche Debugging-Ausgabe
                        console.log('Table height fixed:', tableWrapper.style.height);
                    }
                }
            }
            setInterval(fixTableHeight, 500);  // Wiederhole alle 500ms
            fixTableHeight();  // Sofortige Ausführung
            """
        )
        
        def on_table_select(evt: gr.SelectData, df):
            """Behandelt Zeilenauswahl und Audiovorschau"""
            if evt.index[0] is not None:
                selected_row = df.iloc[evt.index[0]]
                return selected_row['Audio'], evt.index[0]
            return None, None
        
        def move_entry(df, selected_index, direction):
            """
            Verschiebt Einträge robust
            
            Args:
                df (pd.DataFrame): Aktueller DataFrame
                selected_index (int): Index der ausgewählten Zeile
                direction (str): 'up' oder 'down'
            """
            if selected_index is None or selected_index < 0:
                return df
            
            # Kopiere DataFrame, um Seiteneffekte zu vermeiden
            df_copy = df.copy()
            
            if direction == 'up' and selected_index > 0:
                # Tausche Zeilen
                df_copy.iloc[selected_index-1], df_copy.iloc[selected_index] = \
                    df_copy.iloc[selected_index].copy(), df_copy.iloc[selected_index-1].copy()
                selected_index -= 1
            elif direction == 'down' and selected_index < len(df) - 1:
                # Tausche Zeilen
                df_copy.iloc[selected_index+1], df_copy.iloc[selected_index] = \
                    df_copy.iloc[selected_index].copy(), df_copy.iloc[selected_index+1].copy()
                selected_index += 1
            
            # Aktualisiere Ranking
            df_copy['Ranking'] = range(1, len(df_copy) + 1)
            
            return df_copy, selected_index
        
        def scan_speaker_files(directory):
            """Scannt Sprecherdateien"""
            speaker_files = []
            for filename in sorted(os.listdir(directory)):
                if filename.endswith('_1.wav'):
                    match = re.match(r'(\w+)_(\w+)_1\.wav', filename)
                    if match:
                        vorname, name = match.groups()
                        speaker_files.append({
                            'Audio': os.path.join(directory, filename),
                            'Vorname': vorname,
                            'Name': name,
                            'Ranking': len(speaker_files) + 1
                        })
            
            return pd.DataFrame(speaker_files).sort_values(['Vorname', 'Name'])
        
        def save_current_ranking(df):
            """Speichert das aktuelle Ranking"""
            ranking_file = os.path.join(speaker_dir, 'speaker_ranking.csv')
            df.to_csv(ranking_file, index=False)
            return f"Ranking gespeichert in {ranking_file}"
        
        # Event-Handler für Tabellenauswahl
        speaker_table.select(
            fn=on_table_select,
            inputs=[speaker_table],
            outputs=[audio_preview, selected_row_state]
        )
        
        # Event-Handler für Verschieben
        move_up_btn.click(
            fn=lambda df, idx: move_entry(df, idx, 'up'),
            inputs=[speaker_table, selected_row_state],
            outputs=[speaker_table, selected_row_state]
        )
        
        move_down_btn.click(
            fn=lambda df, idx: move_entry(df, idx, 'down'),
            inputs=[speaker_table, selected_row_state],
            outputs=[speaker_table, selected_row_state]
        )
        
        refresh_btn.click(
            fn=lambda: scan_speaker_files(speaker_dir),
            outputs=[speaker_table]
        )
        
        save_ranking_btn.click(
            fn=save_current_ranking,
            inputs=[speaker_table],
            outputs=[gr.Textbox(label="Status")]
        )
        
        # Initial-Laden der Tabelle
        demo.load(
            fn=lambda: scan_speaker_files(speaker_dir),
            outputs=[speaker_table]
        )
