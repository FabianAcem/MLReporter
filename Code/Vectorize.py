from gensim.models import FastText
from nltk.corpus import stopwords
import spacy  # type: ig
from langdetect import detect, detect_langs
import json
import re
import numpy as np
import os


def lade_json(filepath):
    """
    Lädt JSON-Daten aus einer Datei. Gibt eine leere Liste zurück, wenn die Datei nicht existiert oder leer ist.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Überprüfe, ob die geladenen Daten eine Liste sind, sonst gib eine leere Liste zurück.
            # Manchmal ist die JSON-Struktur ein Dictionary, das eine Liste enthält (z.B. {"items": [...]}).
            # Passe dies an deine tatsächliche JSON-Struktur an.
            # Wenn StarredArticles.json oder suggested_articles_for_web.json nur eine Liste von IDs ist:
            if isinstance(data, list):
                return data
            # Wenn es ein Dictionary ist, das die Artikel als Wert eines Schlüssels enthält (wie in deiner initialen JSON):
            elif isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                return data["items"] # Gibt die Liste der Artikel zurück
            # Wenn es sich um eine JSON-Datei handelt, die einfach eine Liste von IDs ist (wie für suggested_articles_for_web.json)
            # und das direkte Laden eine Liste ergibt.
            else:
                return data # Nimmt an, dass die Struktur korrekt ist (z.B. direkt eine Liste von IDs)

    except FileNotFoundError:
        print(f"Warnung: Datei '{filepath}' nicht gefunden. Gibt leere Liste zurück.")
        return []
    except json.JSONDecodeError:
        print(f"Warnung: Datei '{filepath}' ist keine gültige JSON-Datei oder ist leer. Gibt leere Liste zurück.")
        return []
    except Exception as e:
        print(f"Fehler beim Laden von JSON aus '{filepath}': {e}. Gibt leere Liste zurück.")
        return []

def speicher_json(filepath, data):
    """
    Speichert Daten als JSON in einer Datei.
    """
    try:
        # Sicherstellen, dass der Verzeichnispfad existiert
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Fehler beim Speichern von JSON nach '{filepath}': {e}")


lemmatizer = spacy.load("de_core_news_sm")


def is_english(text, threshold=0.9):
    try:
        langs = detect_langs(text)
        for lang in langs:
            if lang.lang == "en" and lang.prob > threshold:
                return True
    except:
        pass
    return False


def deleteenglishArticle(title, summary, max_english_ratio=0.3):
    text_parts = [title, summary]
    englischparts = sum(is_english(part) for part in text_parts)
    return (englischparts / len(text_parts)) > max_english_ratio


def lemmatizeanddeletestops(file):
    SimpleData = lade_json(file)
    stopwort = (
        set(stopwords.words("german"))
        | set(stopwords.words("english"))
        | {"--", "Aussteller-Stimmen", "Fakuma", "2024", "2023", "gleichzeitig", ""}
    )

    PolyKunden = (
        "Best Plastics Management",
        "Condor Compounds",
        "De Paauw",
        "Hexpol",
        "PolymerChemie",
        "IdePro",
        "InnoPlastics",
        "Rissland",
    )
    allentrys = []
    Starredlist = []
    for Eintrag in SimpleData:
        title = Eintrag.get("title","")
        summary = Eintrag.get("summary","")

        title = re.sub(r"\d+", "", title)
        summary = re.sub(r"\d+", "", summary)

        SumList = re.findall(r"\b[\w-]+\b", summary)
        TitelList = re.findall(r"\b[\w-]+\b", title)

        kundegefunden = False
        for kunde in PolyKunden:
            pattern = re.compile(rf"\b{re.escape(kunde)}\b", re.IGNORECASE)
            if pattern.search(Eintrag["title"]) or pattern.search(Eintrag["summary"]):
                print("Treffer: ", kunde, Eintrag["id"])
                Starredlist.append(Eintrag["id"])
                kundegefunden = True
                break

        if kundegefunden:
            continue

        titelText = " ".join(TitelList)
        SumText = " ".join(SumList)

        LemaSum = lemmatizer(SumText)
        LemaTitel = lemmatizer(
            titelText
        )  # das Modell von Spacy erstellt ein Objekt welches eigenschaften zu jedem Wort bereitstellt

        titlelemma = [word.lemma_ for word in LemaTitel]
        sumlemma = [
            word.lemma_ for word in LemaSum
        ]  # Nutzen davon tun wir nur das Lemmatisierte

        Titelnonstop = [wort for wort in titlelemma if wort.lower() not in stopwort]
        Sumnonstop = [wort for wort in sumlemma if wort.lower() not in stopwort]

        finaltitel = " ".join(Titelnonstop)
        Sumfinal = " ".join(Sumnonstop)

        if not deleteenglishArticle(titelText, SumText):
            Eintrag["title"] = finaltitel
            Eintrag["summary"] = Sumfinal
            allentrys.append(Eintrag)

    speicher_json("/root/project/MLReporter/Data/StarredArticles.json", Starredlist)

    return allentrys


insgesamt = []

# insgesamt.extend(
#    lemmatizeanddeletestops("C:/Abschlussprojekt/Code/convertedArticles.json")
# )
# insgesamt.extend(lemmatizeanddeletestops("C:/LF12Project/Scraper/scraperesults.json"))

# speicher_json("C:/Abschlussprojekt/Code/corpus.json", insgesamt)

# all_entrys = lemmatizeanddeletestops(
#    "C:/LF12Project/Scraper/scraperesults.json", all_entrys
# )


def createfasttext():
    Article = lade_json("/root/project/MLReporter/Data/corpus.json")
    corpus = []
    for item in Article:
        titleandsum = f'{item["title"]} {item["summary"]}'.lower().split()
        corpus.append(titleandsum)

    model = FastText(vector_size=100, window=5, workers=4, min_count=2, sg=1, epochs=10)

    model.build_vocab(corpus)

    model.train(corpus, total_examples=len(corpus), epochs=model.epochs)

    model.save("/root/project/MLReporter/Data/Volume1")
    print(model.wv.most_similar("Kunststoff", topn=10))


def vectorize(path):
    model = FastText.load("/root/project/MLReporter/Data/Volume1")
    Article = lade_json(path)
    for item in Article:
        titlevecs = [
            model.wv[word] for word in item["title"].split() if word in model.wv
        ]
        titleall = (
            np.sum(titlevecs, axis=0) if titlevecs else np.zeros(model.vector_size)
        )

        sumvecs = [
            model.wv[word] for word in item["summary"].split() if word in model.wv
        ]
        sumall = np.sum(sumvecs, axis=0) if sumvecs else np.zeros(model.vector_size)

        allvectors = np.concatenate([titleall, sumall])
        item["allinonevector"] = allvectors.tolist()
        del item["summary"]
        del item["title"]
    speicher_json(path, Article)


model = FastText.load("/root/project/MLReporter/Data/Volume1")
if "kunststoff" in model.wv:
    vector = model.wv["kunststoff"]
    print(vector)
else:
    None
# path = "C:/Abschlussprojekt/Code/vectorizedarticles.json"
# speicher_json(path, vectorizeedArticle)
