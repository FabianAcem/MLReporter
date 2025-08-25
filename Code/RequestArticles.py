import requests
import json
import time
import os
from datetime import datetime, timedelta

# Pfad zur Token-Datei und anderen Daten relativ zum Skript-Verzeichnis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
token_datei = os.path.join(DATA_DIR, "token.json")


def lade_token():
    with open(token_datei, "r") as f:
        return json.load(f)


def speicher_token(data):
    with open(token_datei, "w") as f:
        json.dump(data, f, indent=4)


def accesstoken_erneuern():
    tokenset = lade_token()

    if time.time() * 1000 >= tokenset.get("token_expiry", 0):
        print("Token abgelaufen... neuer wird angefordert")

        url = "https://www.inoreader.com/oauth2/token"
        data = {
            "client_id": tokenset["client_id"],
            "client_secret": tokenset["client_secret"],
            "grant_type": "refresh_token",
            "refresh_token": tokenset["refresh_token"],
        }

        response = requests.post(url, data=data)
        if response.status_code == 200:
            neue_tokens = response.json()
            tokenset["access_token"] = neue_tokens["access_token"]
            tokenset["token_expiry"] = time.time() + neue_tokens.get("expires_in", 3600)
            speicher_token(tokenset)
            print("Neuer Token gespeichert")
        else:
            print("Fehler beim Token-Refresh", response.text)

    return tokenset["access_token"]


def TaggedArticleAnfrage(headers):

    articles = []
    taggedurl = "https://www.inoreader.com/reader/api/0/stream/contents?s=user/-/state/com.google/tags&n=100"

    while True:
        response = requests.get(taggedurl, headers=headers)
        if response.status_code == 200:
            rawdata = response.json()
            for item in rawdata["items"]:
                if "user/1005503800/state/com.google/starred" not in item["categories"]:
                    articles.append(item)
            uncontinuation = rawdata.get("continuation")
            if uncontinuation:
                print("Nächsten 100 Artikel werden abgerufen")
                taggedurl = f"https://www.inoreader.com/reader/api/0/stream/contents?s=user/-/state/com.google/tags&n=100&c={uncontinuation}"
            else:
                break
        else:
            print(f"Fehler beim Abrufen der nächsten Seite {response.text}")

    return articles


def FeedArticleRequest(headers, loops=5):

    feedurl = "https://www.inoreader.com/reader/api/0/stream/contents?s=user/-/label/Poly:Reporter&n=100"

    articles = []

    x = 0
    while x <= loops:
        response = requests.get(feedurl, headers=headers)
        if response.status_code == 200:
            rawdata = response.json()
            for item in rawdata["items"]:
                if "user/1005503800/state/com.google/starred" not in item["categories"]:
                    articles.append(item)
            uncontinuation = rawdata.get("continuation")
            if uncontinuation:
                print("Nächsten 100 Artikel werden abgerufen")
                feedurl = f"https://www.inoreader.com/reader/api/0/stream/contents?s=user/-/label/Poly:Reporter&n=100&c={uncontinuation}"
            else:
                break
            x = x + 1

    return articles


def FeedLastWeekRequest(headers):
    eine_Woche = datetime.utcnow() - timedelta(days=7)
    timestamp = int(eine_Woche.timestamp())
    feedurl = "https://www.inoreader.com/reader/api/0/stream/contents"
    paramsurl = {
        "s": "user/-/state/com.google/all_articles",
        "n": 100,
        "ot": timestamp,
        "r": "n",
        "ck": int(datetime.utcnow().timestamp()),
    }
    articles = []
    continuation = None
    max_runs = 20

    for x in range(max_runs):
        if continuation:
            paramsurl["c"] = continuation
        else:
            paramsurl.pop("c", None)

        print("Hole Artikel...")
        response = requests.get(feedurl, headers=headers, params=paramsurl)

        if response.status_code != 200:
            print(f"Fehler: {response.status_code}")
            break
        rawdata = response.json()
        items = rawdata.get("items", [])

        if not items:
            print("Keine weiteren Artikel")

        for item in items:
            published = item.get("published")
            pub_date = datetime.utcfromtimestamp(published)
            if pub_date >= eine_Woche:
                articles.append(item)

        continuation = rawdata.get("continuation")

        if not continuation:
            print("Alle Artikle wurden geladen")
            break

    return articles


def StarredArticleRequest(headers):

    starredurl = "https://www.inoreader.com/reader/api/0/stream/contents?s=user/-/state/com.google/starred&n=100"

    articles = []

    while True:
        response = requests.get(starredurl, headers=headers)
        if response.status_code == 200:
            rawdata = response.json()
            for item in rawdata["items"]:
                articles.append(item)
            incontinuation = rawdata.get("continuation")
            if incontinuation:
                print("Nächsten 100 Artikel werden abgerufen")
                starredurl = f"https://www.inoreader.com/reader/api/0/stream/contents?s=user/-/state/com.google/starred&n=100&c={incontinuation}"
            else:
                break
        else:
            print(f"Fehler beim Abrufen der nächsten Seite {response.text}")

    return articles


if __name__ == "__main__":
    token = accesstoken_erneuern()
    headers = {
        "Authorization": f"Bearer {token}",
        "AppId": "1000001959",
        "AppKey": "rPeJRCeHO0dzYJmIutOmODqrB6DyuKsb",
    }


def sammelArtikel():
    feedandtaggedArticles = TaggedArticleAnfrage(headers)
    feedandtaggedArticles.extend(FeedArticleRequest(headers))
    starredarticle = StarredArticleRequest(headers)

    alle_artikel = feedandtaggedArticles + starredarticle
    output_path = os.path.join(DATA_DIR, "response.json")
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump({"items": alle_artikel}, fp, indent=4, ensure_ascii=False)
