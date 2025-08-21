# app.py
from flask import Flask, render_template, request, redirect, url_for, flash
import json
import os
import pickle # Importiere pickle, um das Modell zu laden

# Importiere deine Funktionen aus Clustering.py und Learning.py
from Clustering import mark_as_starred_inoreader
from Learning import predictArticles as generate_predictions
from Vectorize import lade_json, speicher_json

app = Flask(__name__)
app.secret_key = 'your_secret_key' # Ersetzen Sie dies durch einen sicheren Schlüssel!

ARTICLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')

# --- MODELL EINMALIG LADEN ---
# Dies wird ausgeführt, wenn die app.py das erste Mal geladen wird.
try:
    with open(os.path.join(ARTICLES_DIR, "RF_best_model.pkl"), "rb") as f:
        # Speichere das geladene Modell in einer globalen Variable
        # oder mache es über einen Kontext verfügbar, der für predictArticles zugänglich ist.
        # Für einfache Zwecke speichern wir es hier als Modul-Variable.
        loaded_ml_model = pickle.load(f)
    print("Machine Learning Modell erfolgreich geladen.")
except FileNotFoundError:
    loaded_ml_model = None
    print(f"Fehler: RF_best_model.pkl wurde nicht gefunden unter {os.path.join(ARTICLES_DIR, 'RF_best_model.pkl')}")
    print("Bitte stellen Sie sicher, dass das Modell trainiert und im 'Data'-Ordner gespeichert wurde.")
except Exception as e:
    loaded_ml_model = None
    print(f"Fehler beim Laden des Machine Learning Modells: {e}")
# --- ENDE MODELL LADEN ---


@app.route('/')
def index():
    """Zeigt die vom Modell vorgeschlagenen Artikel an."""
    # ... (restlicher Code wie gehabt) ...
    try:
        suggested_articles_ids = lade_json(os.path.join(ARTICLES_DIR, "suggested_articles_for_web.json"))
        all_current_articles = lade_json(os.path.join(ARTICLES_DIR, "ConvertedCurrent.json"))
        
        articles_to_display = [article for article in all_current_articles if article.get("id") in suggested_articles_ids]
        
        for article in articles_to_display:
            article['selected'] = False

    except FileNotFoundError:
        articles_to_display = []
        flash("Keine Artikelvorschläge gefunden. Bitte generieren Sie Vorhersagen.")
    
    return render_template('index.html', articles=articles_to_display)

@app.route('/submit_starred', methods=['POST'])
def submit_starred():
    """Verarbeitet die vom Benutzer markierten Artikel."""
    # ... (restlicher Code wie gehabt) ...
    selected_article_ids = request.form.getlist('selected_articles')
    
    if selected_article_ids:
        mark_as_starred_inoreader(selected_article_ids)
        flash(f"{len(selected_article_ids)} Artikel wurden erfolgreich als 'später lesen' markiert.")
    else:
        flash("Keine Artikel ausgewählt.")
        
    return redirect(url_for('index'))

@app.route('/request_new_predictions')
def request_new_predictions():
    """Fordert neue Vorhersagen vom Modell an und schließt bereits markierte Artikel aus."""
    if loaded_ml_model is None:
        flash("Modell ist nicht geladen. Kann keine neuen Vorhersagen generieren.")
        return redirect(url_for('index'))

    try:
        articles_to_exclude = lade_json(os.path.join(ARTICLES_DIR, "user_starred_for_exclusion.json"))
    except FileNotFoundError:
        articles_to_exclude = []
    
    # Rufe generate_predictions auf und übergebe das geladene Modell, falls es die Funktion akzeptiert.
    # WICHTIG: Deine 'predictArticles' Funktion (generate_predictions) muss angepasst werden,
    # um das bereits geladene Modell als Parameter zu akzeptieren, statt es selbst zu laden.
    generate_predictions(
        path="ConvertedCurrent.json",
        articles_to_exclude=articles_to_exclude,
        model=loaded_ml_model # Diesen Parameter muss predictArticles nun akzeptieren
    )
    
    flash("Neue Artikelvorschläge wurden generiert.")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False)