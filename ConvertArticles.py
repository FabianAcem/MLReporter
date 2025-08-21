import json
from bs4 import BeautifulSoup
import re
import html
import pandas as pd
from Vectorize import speicher_json
from Vectorize import lade_json


def get_important_info(path, sourcepath):
    with open(sourcepath, encoding="utf-8") as file:
        RawData = json.load(file)
        convertedArticles = []
        Artikel = RawData["items"]
        for item in Artikel:
            convertedArticles.append(
                {
                    "title": item["title"],
                    "id": item["id"],
                    "summary": item["summary"]["content"] if "summary" in item else "",
                    "source": item["origin"]["title"],
                    "length": len(item["summary"]["content"]),
                    "interesting": (
                        1
                        if "user/1005503800/state/com.google/starred"
                        in item["categories"]
                        else 0
                    ),
                    "Bild?": 1 if "enclosure" in item else 0,
                }
            )

    with open(path, "w", encoding="utf-8") as file:
        json.dump(convertedArticles, file, ensure_ascii=False, indent=4)


def get_cleaned_text(path):
    with open(path, encoding="utf-8") as file:
        Article_list = json.load(file)
        for Article in Article_list:
            Summary = Article["summary"]
            Titel = Article["title"]
            Titel = html.unescape(Titel)
            Summary = re.sub(r"\s+", " ", Summary).strip()
            if "<" in Summary or ">" in Summary:
                try:
                    text = BeautifulSoup(Summary, "html.parser")
                    cleandtext = text.get_text(separator=" ").strip()
                except:
                    cleandtext = Summary.strip()
            else:
                cleandtext = Summary.strip()

            taginklammern = re.compile(r"(.*?)\s*\((.*?)\)$")
            source = Article["source"]
            match = taginklammern.match(source)
            if match:
                Source = match.group(1).strip()
                tag = match.group(2).strip()

            else:
                Source = source
                tag = None

            Article["source"] = Source
            Article["tag"] = tag
            Article["summary"] = cleandtext
            Article["title"] = Titel
    with open(path, "w", encoding="utf-8") as file:
        json.dump(Article_list, file, ensure_ascii=False, indent=4)


def hotencoding(path):
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)

    sourceitems = set(item["source"] for item in articles if item["source"] is not None)
    listsources = list(sourceitems)
    listsources.sort()

    tagitems = set(item["tag"] for item in articles if item["tag"] is not None)
    listtags = list(tagitems)
    listtags.sort()

    somethinginsource = lade_json("/root/project/MLReporter/Data/source_map.json")
    if somethinginsource:
        source_map = somethinginsource
    else:
        source_map = {source: f"source_{i}" for i, source in enumerate(listsources)}
        speicher_json("/root/project/MLReporter/Data/source_map.json", source_map)

    somethingintag = lade_json("/root/project/MLReporter/Data/tag_map.json")
    if somethingintag:
        tag_map = somethingintag
    else:
        tag_map = {tag: f"tag_{i}" for i, tag in enumerate(listtags)}
        speicher_json("/root/project/MLReporter/Data/tag_map.json", tag_map)

    for article in articles:
        original_source = article.get("source")
        if original_source in source_map:
            article["source"] = source_map[original_source]
        else:
            article["source"] = "source unknown"

        original_tag = article.get("tag")
        if original_tag in tag_map:
            article["tag"] = tag_map[original_tag]
        else:
            article["tag"] = "tag unknown"

    df = pd.DataFrame(articles)

    all_sources = list(source_map.values()) + ["source unknown"]
    df["source"] = pd.Categorical(df["source"], categories=all_sources)
    sourcehot = pd.get_dummies(df["source"], prefix="", prefix_sep="", dtype=int)

    all_tags = list(tag_map.values()) + ["tag unknown"]
    df["tag"] = pd.Categorical(df["tag"], categories=all_tags)
    taghot = pd.get_dummies(df["tag"], prefix="", prefix_sep="", dtype=int)

    result = pd.concat([df.drop(["source", "tag"], axis=1), sourcehot, taghot], axis=1)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict("records"), f, ensure_ascii=False, indent=4)


path = "/root/project/MLReporter/Data/convertedArticles.json"
sourcepath = "/root/project/MLReporter/Data/response.json"
# get_important_info(path, sourcepath)
# get_cleaned_text(path)
# convertsourcandtag(path)
