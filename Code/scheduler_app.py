# scheduler_app.py
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pickle # Neu: Importiere pickle

# Importiere deine notwendigen Funktionen
from Clustering import ConvertCurrent # Diese Funktion muss angepasst werden
from Vectorize import lade_json # Zum Laden der vorgeschlagenen Artikel

# E-Mail-Konfiguration
EMAIL_ADDRESS = 'Fabian.Acem@kds-software.com'
EMAIL_PASSWORD = 'Nelzu2410+'
RECEIVER_EMAILS = ['Fabian.Acem@kds-software.com']
WEBSITE_LINK = 'http://127.0.0.1:5000'

# Basis-Datenpfad (wie in app.py und Learning.py)
BASE_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')

# --- MODELL EINMALIG LADEN FÜR DEN SCHEDULER ---
loaded_ml_model_for_scheduler = None
try:
    model_filepath = os.path.join(BASE_DATA_PATH, "RF_best_model.pkl")
    with open(model_filepath, "rb") as f:
        loaded_ml_model_for_scheduler = pickle.load(f)
    print("Machine Learning Modell für den Scheduler erfolgreich geladen.")
except FileNotFoundError:
    print(f"Fehler: RF_best_model.pkl wurde nicht gefunden unter {model_filepath}")
    print("Bitte stellen Sie sicher, dass das Modell trainiert und im 'Data'-Ordner gespeichert wurde.")
except Exception as e:
    print(f"Fehler beim Laden des Machine Learning Modells für den Scheduler: {e}")
# --- ENDE MODELL LADEN ---

# In scheduler_app.py

# ... (existing imports and definitions) ...

def send_email_with_articles():
    """Sendet eine E-Mail mit dem Link zur Artikelauswahl."""
    
    if loaded_ml_model_for_scheduler is None:
        print("Modell ist nicht geladen. Überspringe Artikelverarbeitung und E-Mail-Versand.")
        return

    print("Starte wöchentliche Artikelverarbeitung und Vorhersage...")
    ConvertCurrent(model=loaded_ml_model_for_scheduler)
    print("Artikelverarbeitung abgeschlossen.")

    # --- MODIFIZIERTER TEIL HIER ---
    # Initialisiere MIMEMultipart VOR dem try-except Block
    msg = MIMEMultipart("alternative")
    msg['Subject'] = "Deine neuen personalisierten Artikelvorschläge sind da! 🚀"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = ", ".join(RECEIVER_EMAILS)
    # --- ENDE MODIFIZIERTER TEIL ---

    try:
        suggested_ids = lade_json(os.path.join(BASE_DATA_PATH, "suggested_articles_for_web.json"))
        article_count = len(suggested_ids)
    except FileNotFoundError:
        article_count = 0
        print(f"Warnung: Datei '{os.path.join(BASE_DATA_PATH, 'suggested_articles_for_web.json')}' nicht gefunden. Artikelanzahl ist 0.")
    except Exception as e:
        article_count = 0
        print(f"Warnung: Fehler beim Laden von suggested_articles_for_web.json: {e}. Artikelanzahl ist 0.")


    text = f"""Hallo,

Deine neuen personalisierten Artikelvorschläge ({article_count} Artikel) sind verfügbar!

Klicke auf den folgenden Link, um sie anzusehen und deine Favoriten zu markieren:
{WEBSITE_LINK}

Viel Spaß beim Lesen!

Dein Newsletter-Team
"""
    html = f"""\
<html>
  <body>
    <p>Hallo,</p>
    <p>Deine neuen personalisierten Artikelvorschläge ({article_count} Artikel) sind da! 🚀</p>
    <p>Klicke auf den folgenden Link, um sie anzusehen und deine Favoriten zu markieren:</p>
    <p><a href="{WEBSITE_LINK}">Zu deinen Artikeln</a></p>
    <p>Viel Spaß beim Lesen!</p>
    <p>Dein Newsletter-Team</p>
  </body>
</html>
"""
    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')
    msg.attach(part1)
    msg.attach(part2)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, RECEIVER_EMAILS, msg.as_string())
        print("E-Mail erfolgreich gesendet!")
    except Exception as e:
        print(f"Fehler beim Senden der E-Mail: {e}")

# ... (restlicher scheduler_app.py code) ...
if __name__ == '__main__':
    scheduler = BlockingScheduler()
    scheduler.add_job(send_email_with_articles, IntervalTrigger(seconds=60), id='weekly_article_update') # Testzwecke
    
    print("Scheduler gestartet. Die E-Mail wird gemäß Plan gesendet.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass