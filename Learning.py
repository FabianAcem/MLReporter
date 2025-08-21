import os
import pandas as pd
import numpy as np
import pickle
import requests
import matplotlib.pyplot as plt
import json
from gensim.models import FastText
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier # Hier nur einmal definieren
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, recall_score, make_scorer, precision_score, f1_score, roc_auc_score
import warnings
from typing import List, Optional, Tuple, Any

from Vectorize import lade_json # Deine eigenen Module sollten auch hier oben stehen
from Vectorize import speicher_json
data = pd.read_json("/root/project/MLReporter/Data/baseconverted.json")

# Warnungen unterdrücken, um die Ausgabe übersichtlicher zu halten
warnings.filterwarnings('ignore')

def saveForest():
    print("Starte Modelltraining und -optimierung mit RandomizedSearchCV...")

    # 1. Daten laden
    try:
        data = pd.read_json("/root/project/MLReporter/Data/baseconverted.json")
        print("Daten erfolgreich geladen. ✔️")
    except FileNotFoundError:
        print("Fehler: 'baseconverted.json' nicht gefunden. Bitte überprüfe den Pfad. ❌")
        return

    # 2. Daten mischen und aufteilen
    # 'frac=1' mischt die gesamten Daten, 'random_state' sorgt für Reproduzierbarkeit
    shuffleddata = data.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Daten geladen: {shuffleddata.shape[0]} Zeilen, {shuffleddata.shape[1]} Spalten.")

    # Zielvariable (y) und Features (x) definieren
    y = shuffleddata["interesting"]
    x = shuffleddata.drop(["interesting", "id"], axis=1)
    print("Zielvariable und Features definiert. Zielspalte 'interesting' entfernt. 🎯")

    # 3. Feature Engineering: Erstellung neuer Interaktionsmerkmale
    # Diese Interaktionen können dem Modell helfen, komplexere Beziehungen in den Daten zu erkennen.
    print("Starte Feature Engineering... 🛠️")
    x["length_sum"] = x["length"] * x["sum"]
    x["length_avr"] = x["length"] * x["average"]
    x["sum_avr"] = x["sum"] * x["average"]
    x["lengthpic"] = x["Bild?"] * x["length"]
    x["cluster0with"] = x["Cluster"] * x["length"]
    x["cluster1with"] = x["Cluster"] * x["sum"]
    x["cluster2with"] = x["Cluster"] * x["average"]
    x["cluster3with"] = x["Cluster"] * x["Bild?"]

    # Weitere mögliche Feature-Interaktionen (Beispiele):
    # Du kannst hier weitere logische Kombinationen deiner Features hinzufügen,
    # die für dein Problem sinnvoll erscheinen.
    x["sum_ratio_length"] = x["sum"] / (x["length"] + 1e-6) # Vermeide Division durch Null
    x["average_plus_sum"] = x["average"] + x["sum"]
    x["length_squared"] = x["length"] ** 2
    x["sum_squared"] = x["sum"] ** 2
    x["bild_and_avg"] = x["Bild?"] * x["average"]
    x["cluster_times_bild"] = x["Cluster"] * x["Bild?"]
    print(f"Feature Engineering abgeschlossen. Neue Spalten: {x.shape[1]}")

    # 4. Aufteilung in Trainings- und Testsets
    # 'test_size=0.20' bedeutet 20% der Daten werden für das Testen verwendet.
    # 'random_state' sorgt für Reproduzierbarkeit der Aufteilung.
    # 'stratify=y' ist entscheidend, um die Klassenverteilung der Zielvariablen in Trainings- und Testset zu erhalten.
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.20, random_state=101, stratify=y
    )
    print(f"Daten aufgeteilt: X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape} 📊")

    # 5. Definition des Hyperparameter-Rasters für die Optimierung
    # Erweiterte Parameter für eine umfassendere Suche, die von RandomizedSearchCV effizient erkundet wird.
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)] # Mehr Bäume, breiterer Bereich
    max_features = ['sqrt', 'log2', None] # Verschiedene Strategien für Features pro Split
    max_depth = [10, 20, 30, None] # 'None' bedeutet unbegrenzte Tiefe
    min_samples_split = [2, 5, 10, 20]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    criterion = ["gini", "entropy"]
    class_weight = [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, "balanced"] # Mehr Optionen für Klassenungleichgewicht

    # Das Parameter-Gitter für RandomizedSearchCV
    param_distributions = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
        "criterion": criterion,
        "class_weight": class_weight,
    }
    print("Hyperparameter-Gitter für RandomizedSearchCV definiert. ⚙️")

    # 6. Definition der Scoring-Metrik
    # Da du Recall als wichtig erachtest, behalten wir diesen als primäre Metrik.
    # Du könntest hier auch 'f1_score' oder 'roc_auc_score' verwenden, je nach Priorität.
    scorer = make_scorer(recall_score, pos_label=1)
    print("Scoring-Metrik (Recall) definiert. 📈")

    # 7. Hyperparameter-Optimierung mit RandomizedSearchCV
    # RandomizedSearchCV führt eine zufällige Suche durch das param_distributions-Gitter durch.
    # 'n_iter' steuert, wie viele verschiedene Parameterkombinationen getestet werden.
    # 'cv=5' erhöht die Robustheit der Cross-Validation.
    # 'n_jobs=-1' nutzt alle verfügbaren CPU-Kerne für schnellere Berechnung.
    print(f"Starte RandomizedSearchCV mit {param_distributions['n_estimators']} Iterationen und {param_distributions['max_features']} Features. Das dauert einen Moment... ⏳")
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42), # random_state für Reproduzierbarkeit des RF
        param_distributions=param_distributions,
        n_iter=100, # Anzahl der Iterationen, für 10k Zeilen ein guter Startpunkt
        scoring=scorer,
        cv=5, # 5-fold Cross-Validation für robustere Bewertung
        n_jobs=-1, # Nutzt alle verfügbaren Kerne
        verbose=2, # Zeigt detaillierteren Fortschritt
        random_state=42 # Für Reproduzierbarkeit der zufälligen Parameter-Auswahl
    )

    random_search.fit(X_train, Y_train)
    print("RandomizedSearchCV abgeschlossen. 🎉")

    # 8. Bestes Modell und seine Performance
    best_estimator = random_search.best_estimator_
    print("\n--- Bestes Modell gefunden ---")
    print(f"Beste Parameter: {random_search.best_params_}")
    print(f"Bester Score (auf Validierungs-Sets): {random_search.best_score_:.4f}")

    # Vorhersagen auf dem Testset mit dem besten Modell
    y_proba = best_estimator.predict_proba(X_test)[:, 1] # Wahrscheinlichkeiten für die positive Klasse
    print("\n--- Evaluierung auf dem Testset ---")
    print("Vorhergesagte Wahrscheinlichkeiten (erste 10):", y_proba[:10])

    # 9. Bewertung mit verschiedenen Schwellenwerten
    # Die Wahl des Schwellenwerts ist entscheidend, besonders bei unbalancierten Klassen oder spezifischen Geschäftsanforderungen.
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        print(f"\n--- Schwellenwert: {threshold:.1f} ---")
        print(classification_report(Y_test, y_pred))
        
        # Zusätzliche Metriken für einen besseren Überblick
        print(f"Recall: {recall_score(Y_test, y_pred, pos_label=1):.4f}")
        print(f"Precision: {precision_score(Y_test, y_pred, pos_label=1):.4f}")
        print(f"F1-Score: {f1_score(Y_test, y_pred, pos_label=1):.4f}")
        
    # Bewertung mit dem standardmäßig optimalen Schwellenwert (oft 0.5, oder basierend auf bester F1/Recall)
    y_pred_final = best_estimator.predict(X_test) # Nutzt den internen Schwellenwert des Modells (meist 0.5)
    print("\n--- Finaler Klassifikationsbericht (Standard-Schwellenwert des Modells) ---")
    print(classification_report(Y_test, y_pred_final))
    print(f"ROC-AUC Score: {roc_auc_score(Y_test, y_proba):.4f}") # AUC-ROC ist eine gute Gesamtmetrik, unabhängig vom Schwellenwert

    # 10. Speichern des besten Modells
    # Speichern des 'best_estimator_' anstelle des gesamten 'random_search' Objekts.
    # Das ist effizienter, da es nur das trainierte Modell und nicht die gesamte Suchhistorie speichert.
    model_filename = "RF_best_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(best_estimator, f)
    print(f"\nBestes Random Forest Modell gespeichert unter: {model_filename} 💾")
    print("Modelltraining und -optimierung abgeschlossen. Viel Erfolg! ✨")

# Beispielaufruf der Funktion (auskommentiert, damit sie nicht automatisch läuft)
#saveForest()


# df= data
# print(data.describe())

# Definieren des Basis-Datenpfades einmalig auf Modulebene
# Dadurch muss dieser Pfad nicht in jeder Funktion neu berechnet werden.
# Der Pfad geht zwei Ebenen nach oben (von Code/ zu Abschlussprojekt/) und dann in den Data/ Ordner.



# Stelle sicher, dass BASE_DATA_PATH korrekt ist:
# /.../MLReporter/Data
BASE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data"
)

def _safe_write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _resolve_path(p: str) -> str:
    """Wenn p relativ ist, interpretiere relativ zu BASE_DATA_PATH; sonst nutze p wie übergeben."""
    return p if os.path.isabs(p) else os.path.join(BASE_DATA_PATH, p)

def predictArticles(
    path: str = "ConvertedCurrent.json",
    articles_to_exclude: Optional[List[Any]] = None,
    model: Optional[Any] = None,
    suggestions_out_path: Optional[str] = None,
    top_k: int = 5
) -> Tuple[bool, str]:
    """
    - path: (abs/rel) Pfad zu ConvertedCurrent.json (Liste von Artikeln mit 'id', ...).
    - articles_to_exclude: IDs, die ausgeschlossen werden sollen (int/str gemischt ok).
    - model: bereits geladener Klassifikator/Regressor; falls None, wird RF_best_model.pkl geladen.
    - suggestions_out_path: (abs/rel) Pfad, wohin [id, id, ...] geschrieben wird (z. B. Data/suggested_articles_for_web.json).
    - top_k: Anzahl gewünschter Vorschläge.
    Rückgabe: (ok, msg)
    """
    try:
        if suggestions_out_path is None:
            return False, "suggestions_out_path not provided"

        suggestions_out_path = _resolve_path(suggestions_out_path)
        data_filepath = _resolve_path(path)

        # 1) Modell bereitstellen
        loaded_model = model
        if loaded_model is None:
            model_filepath = os.path.join(BASE_DATA_PATH, "RF_best_model.pkl")
            try:
                with open(model_filepath, "rb") as f:
                    loaded_model = pickle.load(f)
                print(f"ML-Modell von '{model_filepath}' geladen.")
            except FileNotFoundError:
                _safe_write_json(suggestions_out_path, [])
                return False, f"Modell nicht gefunden unter '{model_filepath}'."
            except Exception as e:
                _safe_write_json(suggestions_out_path, [])
                return False, f"Fehler beim Laden des Modells: {e}"
        else:
            print("Vorhandenes ML-Modell verwendet.")

        # 2) Daten laden
        try:
            df = pd.read_json(data_filepath)
        except FileNotFoundError:
            _safe_write_json(suggestions_out_path, [])
            return False, f"Artikeldatei nicht gefunden unter '{data_filepath}'."
        except Exception as e:
            _safe_write_json(suggestions_out_path, [])
            return False, f"Fehler beim Laden der Artikeldaten: {e}"

        if df.empty:
            _safe_write_json(suggestions_out_path, [])
            return False, "Keine Artikel in ConvertedCurrent.json."

        # 3) Exclusions säubern
        exclude_set = set(str(x) for x in (articles_to_exclude or []))
        if "id" not in df.columns:
            _safe_write_json(suggestions_out_path, [])
            return False, "Spalte 'id' fehlt in den Artikeldaten."

        before = len(df)
        df = df[~df["id"].astype(str).isin(exclude_set)].copy()
        print(f"Artikel vor Ausschluss: {before}, nach Ausschluss: {len(df)}")

        if df.empty:
            _safe_write_json(suggestions_out_path, [])
            return False, "Nach Ausschluss keine Kandidaten mehr."

        # 4) Features sicherstellen
        required_base_cols = ["length", "sum", "average", "Bild?", "Cluster"]
        for col in required_base_cols:
            if col not in df.columns:
                print(f"Warnung: '{col}' fehlt, setze 0.")
                df[col] = 0

        # vorher existierende Features
        df["length_sum"]   = df["length"] * df["sum"]
        df["length_avr"]   = df["length"] * df["average"]
        df["sum_avr"]      = df["sum"] * df["average"]
        df["lengthpic"]    = df["Bild?"] * df["length"]
        df["cluster0with"] = df["Cluster"] * df["length"]
        df["cluster1with"] = df["Cluster"] * df["sum"]
        df["cluster2with"] = df["Cluster"] * df["average"]
        df["cluster3with"] = df["Cluster"] * df["Bild?"]

        # fehlendes Feature lengthandsum (steht in feature_columns, also bauen!)
        if "lengthandsum" not in df.columns:
            df["lengthandsum"] = df["length"] + df["sum"]

        # neuere Features aus deinem Code
        df["average_plus_sum"] = (df["average"] + df["sum"]) if ("average" in df.columns and "sum" in df.columns) else 0
        df["bild_and_avg"]     = (df["Bild?"] * df["average"]) if ("Bild?" in df.columns and "average" in df.columns) else 0
        df["cluster_times_bild"] = (df["Cluster"] * df["Bild?"]) if ("Cluster" in df.columns and "Bild?" in df.columns) else 0
        df["length_squared"]     = (df["length"] ** 2) if ("length" in df.columns) else 0
        if "sum" in df.columns and "length" in df.columns:
            df["sum_ratio_length"] = df.apply(lambda r: (r["sum"] / r["length"]) if r["length"] != 0 else 0, axis=1)
        else:
            df["sum_ratio_length"] = 0
        df["sum_squared"] = (df["sum"] ** 2) if ("sum" in df.columns) else 0

        # 5) Feature-Reihenfolge (wie trainiert)
        feature_columns = [
            "length", "Bild?",
            "source_0", "source_1", "source_2", "source_3", "source_4", "source_5", "source_6", "source_7", "source_8", "source unknown",
            "tag_0", "tag_1", "tag_2", "tag_3", "tag_4", "tag unknown",
            "Cluster", "average", "sum", "lengthandsum",
            "length_sum", "length_avr", "sum_avr", "lengthpic",
            "cluster0with", "cluster1with", "cluster2with", "cluster3with",
            "average_plus_sum",
            "bild_and_avg",
            "cluster_times_bild",
            "length_squared",
            "sum_ratio_length",
            "sum_squared",
        ]
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
                print(f"Hinweis: fehlende Feature-Spalte '{col}' mit 0 ergänzt.")

        X = df[feature_columns]

        # 6) Scoring
        try:
            if hasattr(loaded_model, "predict_proba"):
                proba = loaded_model.predict_proba(X)
                # falls binäre Klassifikation → Spalte 1 als "Relevanz"
                score = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                df["score"] = score
            else:
                # Regressor oder Klassifikator ohne proba → predict als Score
                pred = loaded_model.predict(X)
                df["score"] = pd.to_numeric(pred, errors="coerce").fillna(0)
        except Exception as e:
            _safe_write_json(suggestions_out_path, [])
            return False, f"Fehler bei der Modellvorhersage: {e}"

        # 7) Top-K IDs bestimmen
        top_k = max(1, int(top_k))
        top_df = df.sort_values("score", ascending=False).head(top_k)
        suggested_ids = top_df["id"].tolist()

        # 8) Schreiben
        _safe_write_json(suggestions_out_path, suggested_ids)
        return True, f"{len(suggested_ids)} Vorschläge nach '{suggestions_out_path}' geschrieben."

    except Exception as e:
        try:
            if suggestions_out_path:
                _safe_write_json(suggestions_out_path, [])
        except:
            pass
        return False, f"Unerwarteter Fehler: {e}"


if __name__ == "__main__":
    import argparse, os
    BASE_DATA_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Data"
    )
    parser = argparse.ArgumentParser(description="Learning tasks")
    parser.add_argument("--train", action="store_true", help="Modell trainieren und speichern")
    parser.add_argument("--train-data", default=os.path.join(BASE_DATA_PATH, "ConvertedCurrent.json"))
    parser.add_argument("--model-out",  default=os.path.join(BASE_DATA_PATH, "RF_best_model.pkl"))
    args = parser.parse_args()

    if args.train:
        train_and_save_forest(args.train_data, args.model_out)
        print("[Done] Training abgeschlossen.")
    else:
        print("Nichts zu tun. Nutze --train, um das Modell zu trainieren.")

