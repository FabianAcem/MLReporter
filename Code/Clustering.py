import pandas as pd
import matplotlib.pyplot as plt  # type: ignore
import json
from gensim.models import FastText  # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pickle
import requests
import os

from RequestArticles import TaggedArticleAnfrage
from RequestArticles import StarredArticleRequest
from RequestArticles import FeedArticleRequest
from RequestArticles import accesstoken_erneuern
from RequestArticles import FeedLastWeekRequest

from ConvertArticles import get_cleaned_text
from ConvertArticles import get_important_info
from ConvertArticles import hotencoding

from Vectorize import lemmatizeanddeletestops
from Vectorize import vectorize
from Vectorize import speicher_json
from Vectorize import lade_json

from Learning import predictArticles
def train_model():
    from Learning import saveForest
    saveForest
# --- Globale Pfadkonfiguration ---
BASE_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')

# --- DEFINIERE HEADERS AUF MODULEBENE ---
# Dies wird einmal ausgeführt, wenn Clustering.py importiert wird.
# Stellen Sie sicher, dass accesstoken_erneuern() hier korrekt funktioniert.
try:
    token = accesstoken_erneuern()
    headers = {
        "Authorization": f"Bearer {token}",
        "AppId": "1000001959",
        "AppKey": "rPeJRCeHO0dzYJmIutOmODqrB6DyuKsb",
    }
    print("API Headers in Clustering.py erfolgreich initialisiert.")
except Exception as e:
    headers = {} # Setze leere Headers, um Fehler zu vermeiden, aber API-Aufrufe werden fehlschlagen
    print(f"Fehler beim Initialisieren der API Headers in Clustering.py: {e}")
# --- ENDE DEFINITION HEADERS ---


def mark_as_starred(): # Dies ist die alte, redundante Funktion. Du hast bereits mark_as_starred_inoreader.
    url = "https://www.inoreader.com/reader/api/0/stream/contents"
    ShouldbeStarred = lade_json(os.path.join(BASE_DATA_PATH, "StarredArticles.json"))
    for predict in ShouldbeStarred:
        payload = {"i": predict, "a": "user/-/label/TRy"} # Überprüfe diesen Tag!

        response = requests.post(
            "https://www.inoreader.com/reader/api/0/edit-tag",
            headers=headers, # Nutze die globalen Headers
            data=payload,
        )

        if response.status_code == 200:
            print(f"Artikel {predict} markiert")
        else:
            print(f"Hat nicht funktioniert zu markieren: {response.text}")


def CurrentArticles():
    # Nutze die globalen Headers für den Request.
    # Da 'headers' jetzt auf Modulebene definiert ist, ist es hier verfügbar.
    current_articles_data = FeedLastWeekRequest(headers)
    print(f"{len(current_articles_data)} Artikel gespeichert")
    filepath = os.path.join(BASE_DATA_PATH, "CurrentArticles.json")
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump({"items": current_articles_data}, fp, indent=4, ensure_ascii=False)


def addclustertoarticle(path):
    kmeans_model_path = os.path.join(BASE_DATA_PATH, "kmeans_model.pkl")
    with open(kmeans_model_path, "rb") as f:
        loaded_means = pickle.load(f)
    kmeans = loaded_means
    
    labels = kmeans.labels_

    allarticles = lade_json(path)

    for index, item in enumerate(allarticles):
        if index < len(labels):
            item["Cluster"] = int(labels[index])
        else:
            item["Cluster"] = -1

    speicher_json(path, allarticles)


def SumAndAvrg(path):
    articlelist = lade_json(path)
    for item in articlelist:
        if "allinonevector" in item and item["allinonevector"] is not None:
            allvectors = np.array(item["allinonevector"])

            average_embedding = np.mean(allvectors)
            sum_embedding = np.sum(allvectors)

            item["average"] = float(average_embedding)
            item["sum"] = float(sum_embedding)

            if "sum" in item and "length" in item:
                lenghthandsum = item["sum"] * item["length"]
                item["lengthandsum"] = lenghthandsum
            else:
                item["lengthandsum"] = 0

            del item["allinonevector"]
        else:
            item["average"] = 0.0
            item["sum"] = 0.0
            item["lengthandsum"] = 0.0
            print(f"Warnung: 'allinonevector' fehlt oder ist leer für Artikel {item.get('id', 'Unbekannt')}")

    speicher_json(path, articlelist)


def scalevectors(path):
    allArticles = lade_json(path)
    
    valid_articles = [item for item in allArticles if "allinonevector" in item and item["allinonevector"]]
    
    if not valid_articles:
        print("Keine gültigen Vektoren zum Skalieren gefunden. Überspringe Skalierung.")
        return

    allvectors = np.array([item["allinonevector"] for item in valid_articles])
    scaledlength = np.log1p([item["length"] for item in valid_articles])

    scaler = StandardScaler()
    scaledvecs = scaler.fit_transform(allvectors)

    updated_allArticles = []
    valid_article_ids = {article['id']: article for article in valid_articles}
    for original_item in allArticles:
        if original_item['id'] in valid_article_ids:
            updated_allArticles.append(valid_article_ids[original_item['id']])
        else:
            updated_allArticles.append(original_item)

    speicher_json(path, updated_allArticles)


def extractforcluster(path):
    holearticles = lade_json(path)
    feature_data = []
    ids = []

    for item in holearticles:
        ids.append(item.get("id", ""))

        length = item.get("length", 0)
        bild = item.get("Bild?", 0)

        meta_data = [length, bild]

        source_feature = [item.get(f"source_{i}", 0) for i in range(9)] + [item.get("source unknown", 0)]
        tag_feature = [item.get(f"tag_{i}", 0) for i in range(5)] + [item.get("tag unknown", 0)]

        vector_feature = item.get("allinonevector", [])
        vector_feature = np.array(vector_feature) if vector_feature else np.zeros(0)

        try:
            feature = np.concatenate(
                [np.array(meta_data), np.array(source_feature), np.array(tag_feature), vector_feature]
            )
            feature_data.append(feature.tolist())
        except ValueError as e:
            print(f"Fehler beim Konkatenieren der Features für Artikel {item.get('id', 'Unbekannt')}: {e}")
            ids.pop()
            continue

    if not feature_data:
        return np.array([]), np.array([]), np.array([])
        
    max_len = max(len(f) for f in feature_data)
    padded_feature_data = np.array([f + [0]*(max_len - len(f)) for f in feature_data])

    labels = np.array([item.get("interesting", 0) for item in holearticles if item.get("id", "") in ids])
    
    return np.array(ids), padded_feature_data, labels


def kmeans(x):
    kmeans_model_path = os.path.join(BASE_DATA_PATH, "kmeans_model.pkl")
    kmeans_instance = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_instance.fit(x)
    with open(kmeans_model_path, "wb") as f:
        pickle.dump(kmeans_instance, f)
    print(f"K-Means Modell unter {kmeans_model_path} gespeichert.")


def lookatclusters(path, x):
    kmeans_model_path = os.path.join(BASE_DATA_PATH, "kmeans_model.pkl")
    Rawfile = lade_json(path)
    with open(kmeans_model_path, "rb") as f:
        loaded_kmeans = pickle.load(f)

    if x.size == 0:
        print("Keine Daten für Cluster-Visualisierung oder Analyse vorhanden.")
        return []

    labels = loaded_kmeans.labels_

    cluster = len(set(loaded_kmeans.labels_))

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(x)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap="viridis")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Visualisierung der Cluster im 2D-Raum")
    plt.colorbar()
    plt.show()

    clustered_articles = {i: [] for i in range(cluster)}
    for idx, label in enumerate(labels):
        if idx < len(Rawfile):
            clustered_articles[label].append(Rawfile[idx])

    for cluster_id, articles in clustered_articles.items():
        interessant = sum(1 for a in articles if a.get("interesting", 0) == 1)
        print(
            f"Cluster{cluster_id}: {len(articles)} Artikel gesamt, davon {interessant} interessant"
        )
    return labels


def perfectkmeans(x, max_k):
    means = []
    inertias = []

    if x.size == 0:
        print("Keine Daten für die Bestimmung der optimalen Clusteranzahl vorhanden.")
        return

    for k in range(1, max_k):
        kmeans_instance = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_instance.fit(x)
        means.append(k)
        inertias.append(kmeans_instance.inertia_)

    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, "o-")
    plt.xlabel("Nummer der Clusters")
    plt.ylabel("Trägheit")
    plt.title("Ellbogen-Methode zur Bestimmung der optimalen Clusteranzahl")
    plt.grid(True)
    plt.show()


def ConvertCurrent(model=None):
    """
    Konvertiert aktuelle Artikel, verarbeitet sie und ruft die Vorhersage auf.
    Diese Funktion wird wöchentlich vom Scheduler aufgerufen.

    Args:
        model (object, optional): Das bereits geladene Machine Learning Modell.
                                  Wird an predictArticles weitergegeben.
    """
    current_articles_sourcepath = os.path.join(BASE_DATA_PATH, "CurrentArticles.json")
    converted_current_path = os.path.join(BASE_DATA_PATH, "ConvertedCurrent.json")

    CurrentArticles()
    get_important_info(converted_current_path, current_articles_sourcepath)
    get_cleaned_text(converted_current_path)
    hotencoding(converted_current_path)
    lemmatizedanddeletedstops = lemmatizeanddeletestops(converted_current_path)
    speicher_json(converted_current_path, lemmatizedanddeletedstops)
    vectorize(converted_current_path)
    scalevectors(converted_current_path)
    addclustertoarticle(converted_current_path)
    SumAndAvrg(converted_current_path)
    
    predictArticles(path=os.path.basename(converted_current_path), model=model)
    # Entferne den direkten Aufruf von mark_as_starred() hier,
    # da die Flask-App dies nach Benutzerinteraktion handhabt.
    # mark_as_starred()


def Basedata():
    """
    Führt die initiale Datenbeschaffung, Vorverarbeitung, Modelltraining (K-Means, Random Forest)
    und Speicherung der Modelle durch.
    Diese Funktion sollte nur einmalig zur Initialisierung/zum Retraining aufgerufen werden.
    """
    print("Starte Basedata: Datenbeschaffung und Modelltraining...")
    sourcepath = os.path.join(BASE_DATA_PATH, "base.json")
    path = os.path.join(BASE_DATA_PATH, "baseconverted.json")

    get_important_info(path, sourcepath)
    get_cleaned_text(path)
    hotencoding(path)
    stemmandstop = lemmatizeanddeletestops(path)
    speicher_json(path, stemmandstop)
    vectorize(path)
    scalevectors(path)
    
    ids, x, y = extractforcluster(path)

    if x.size == 0:
        print("Keine Features für das Clustering/Training gefunden. Basedata wird abgebrochen.")
        return

    perfectkmeans(x, 10)
    kmeans(x)
    lookatclusters(path, x)
    addclustertoarticle(path)
    SumAndAvrg(path)
    train_model()
    print("Basedata abgeschlossen: Modelle trainiert und gespeichert.")


def save_starred_for_exclusion(article_ids):
    """
    Speichert eine Liste von Artikel-IDs, die vom Benutzer als 'starred' markiert wurden.
    Diese Liste wird später verwendet, um bereits markierte Artikel aus neuen Vorschlägen auszuschließen.
    """
    filepath = os.path.join(BASE_DATA_PATH, "user_starred_for_exclusion.json")
    try:
        existing_starred_ids = lade_json(filepath)
    except FileNotFoundError:
        existing_starred_ids = []

    updated_starred_ids = list(set(existing_starred_ids + article_ids))

    speicher_json(filepath, updated_starred_ids)
    print(f"{len(article_ids)} Artikel-IDs zur Ausschlussliste hinzugefügt.")


def mark_as_starred_inoreader(article_ids):
    """
    Markiert Artikel in Inoreader als 'starred' und speichert die IDs zur zukünftigen Exklusion.
    """
    url = "https://www.inoreader.com/reader/api/0/edit-tag"
    # Der Token wird aus der globalen 'headers' Variable verwendet.
    # Du könntest auch accesstoken_erneuern() hier aufrufen, um sicherzustellen, dass es aktuell ist,
    # aber wenn headers auf Modulebene korrekt initialisiert wurden, sollte es passen.
    # token = accesstoken_erneuern()
    # headers = {
    #     "Authorization": f"Bearer {token}",
    #     "AppId": "1000001959",
    #     "AppKey": "rPeJRCeHO0dzYJmIutOmODqrB6DyuKsb",
    # }
    
    successful_starred_ids = []
    for article_id in article_ids:
        # Korrigierter Payload für Inoreader "starred" Status
        payload = {"i": article_id, "a": "user/-/state/com.google/starred"}

        response = requests.post(
            "https://www.inoreader.com/reader/api/0/edit-tag",
            headers=headers, # Nutze die globalen Headers
            data=payload,
        )

        if response.status_code == 200:
            print(f"Artikel {article_id} erfolgreich in Inoreader als 'starred' markiert.")
            successful_starred_ids.append(article_id)
        else:
            print(f"Fehler beim Markieren des Artikels {article_id}: {response.text}")
    
    save_starred_for_exclusion(successful_starred_ids)


def getclusteringdata():
    """
    Holt Artikel zum Clustering/Training und speichert sie in base.json.
    Diese Funktion sollte einmalig ausgeführt werden, um die Basisdaten zu erstellen.
    """
    print("Starte getclusteringdata: Artikel von Inoreader abrufen...")
    # headers müssen hier verfügbar sein, d.h. die globale Initialisierung am Anfang ist wichtig
    feedandtaggedArticles = TaggedArticleAnfrage(headers)
    feedandtaggedArticles.extend(FeedArticleRequest(headers, 3))
    starredarticle = StarredArticleRequest(headers)

    alle_artikel = feedandtaggedArticles + starredarticle

    filepath = os.path.join(BASE_DATA_PATH, "base.json")
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump({"items": alle_artikel}, fp, indent=4, ensure_ascii=False)
    print(f"Basisdaten in {filepath} gespeichert. Gesamtartikel: {len(alle_artikel)}")


# --- Diese Funktion scheint eine Duplikat von mark_as_starred_inoreader zu sein und sollte entfernt werden ---
# Umbenannt, um nicht mit der alten Funktion zu kollidieren.
# Die Logik ist dieselbe wie in mark_as_starred_inoreader, also ist dies redundant.
# Diese Funktion sollte entfernt werden.
def mark_as_starred_old_redundant(headers_param): # Um Namenskollision zu vermeiden
    url = f"https://www.inoreader.com/reader/api/0/edit-tag"
    IDs = lade_json(os.path.join(BASE_DATA_PATH, "StarredArticles.json")) # Korrekter Dateiname
    for article_id in IDs:
        data = {"s": "user/-/state/com.google/starred", "a": "add", "i": article_id}

        response = requests.post(url, headers=headers_param, data=data)

        if response.status_code == 200:
            print(f"Artikel {article_id} erfolgreich als 'starred' markiert!")
        else:
            print(f"Fehler beim Markieren des Artikels {article_id}: {response.text}")
# --- Ende Duplikat-Funktion ---


# --- Steuerung des Skripts, wenn es direkt ausgeführt wird (z.B. für Initialisierung/Training) ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clustering and data initialization tasks")
    parser.add_argument("--init", action="store_true",
                        help="Fetch base data and train all models. Run this only for initialization.")
    parser.add_argument("--fetch-and-cluster", action="store_true",
                        help="Artikel aus Inoreader holen und clustern (placeholder)")
    parser.add_argument("--train", action="store_true",
                        help="ML-Modell trainieren (manuell)")

    args = parser.parse_args()

    if args.init:
        print("--- Starting Full Initialization ---")
        # The initialization process requires fetching data first, then processing and training.
        getclusteringdata()
        Basedata()
        print("--- Full Initialization Complete ---")

    if args.fetch_and-cluster:
        # This is a placeholder from the original code.
        # For a real implementation, one would call specific functions.
        print("[Info] --fetch-and-cluster is a placeholder and has no action.")

    if args.train:
        print("--- Starting Manual Model Training ---")
        train_model()
        print("--- Manual Model Training Complete ---")
