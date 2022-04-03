import ast
import asyncio
import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, datetime
from pathlib import Path
from time import perf_counter
from typing import Final, Collection, List, Any, Hashable

import firebase_admin
import numpy as np
import pandas as pd
import requests
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import CollectionReference, Client
from google.cloud.firestore_v1.types import Document
from pandas import Series
from requests.exceptions import SSLError

STEAMSPY_URL: Final[str] = "https://steamspy.com/api.php"
STORE_STEAMPOWERED_URL: Final[str] = "http://store.steampowered.com/api/appdetails"
INTERESTING_DATA: Final[List[str]] = [
    "type",
    "name",
    "steam_appid",
    "dlc",
    "detailed_description",
    "about_the_game",
    "short_description",
    "header_image",
    "website",
    "pc_requirements",
    "mac_requirements",
    "linux_requirements",
    "developers",
    "publishers",
    "price_overview",
    "platforms",
    "metacritic",
    "reviews",
    "categories",
    "genres",
    "screenshots",
    "recommendations",
    "release_date",
    "background",
    "content_descriptors",
]

PATH_ROOT: Path = Path("D:\\3PycharmProjects\\firebase")
PATH_CREDENTIALS: Final[Path] = PATH_ROOT / "credentials.json"
PATH_DOWNLOAD_DIRECTORY: Final[Path] = PATH_ROOT / "download"
PATH_INDEX: Final[Path] = PATH_DOWNLOAD_DIRECTORY / "index.txt"
PATH_CSV: Final[Path] = PATH_DOWNLOAD_DIRECTORY / "steampowered_app_data.csv"

FORMAT: Final[str] = "%(levelname)s @ %(asctime)s.%(msecs)03d: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def get_request(url: str, params: dict = None) -> dict:
    """Return json-formatted response of a get request using optional parameters."""
    try:
        response = requests.get(url=url, params=params)
    except SSLError as exception:
        log.warning(f"SSL Error: {exception}")

        for i in range(5, 0, -1):
            log.info(f"Waiting... ({i})")
            time.sleep(1)
        log.info("Retrying.")

        return get_request(url=url, params=params)

    if response:
        log.info(f"Response: {response.status_code}")
        return response.json()

    log.warning("No response, waiting 10 seconds...")
    time.sleep(10)
    log.info("Retrying...")
    return get_request(url=url, params=params)


def parse_steam_request(appid: int, name: str, params: dict) -> dict:
    """Unique parser to handle data from Steam Store API."""
    json_data = get_request(url=STORE_STEAMPOWERED_URL, params=params)
    json_app_data = json_data[str(appid)]

    return json_app_data["data"] if json_app_data["success"] else {"name": name, "steam_appid": appid}


def reset_index(filepath: Path) -> None:
    """Reset index in file to 0."""
    with open(filepath, "w", encoding="utf-8") as file:
        file.write("0")


def get_index(filepath: Path) -> int:
    """Retrieve index from file, returning 0 if file not found."""
    try:
        with open(filepath, encoding="utf-8") as file:
            index = int(file.readline())
    except FileNotFoundError:
        index = 0

    return index


def prepare_data_file(filepath: Path, index: int, columns: Collection) -> None:
    """Create file and write headers if index is 0."""
    if index == 0:
        with open(filepath, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()


def process_batches(
        apps_df: pd.DataFrame,
        data_filepath: Path,
        index_filepath: Path,
        columns: List[str],
        begin: int = 0,
        end: int = -1,
        batch_size: int = 100,
        pause: int = 1,
) -> None:
    """Process app data in batches, writing directly to file."""
    log.info(f"Starting at index {begin}:\n")

    if end == -1:
        end = len(apps_df) + 1

    # generate array of batch begin and end points
    batches = np.arange(start=begin, stop=end, step=batch_size)
    batches = np.append(arr=batches, values=end)

    apps_written = 0
    batch_times = []

    for i in range(len(batches) - 1):
        start_time = perf_counter()

        start = batches[i]
        stop = batches[i + 1]

        app_data = get_app_data(apps_df, start, stop, pause)

        # writing app data to file
        with open(data_filepath, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")

            for j in range(3, 0, -1):
                log.info(f"About to write data, don't stop script! ({j})")
                time.sleep(0.5)

            writer.writerows(app_data)
            log.info(f"Exported lines {start}-{stop - 1} to '{data_filepath}'.")

        apps_written += len(app_data)

        # writing last index to file
        with open(index_filepath, "w", encoding="utf-8") as file:
            file.write(str(stop))

        time_taken = perf_counter() - start_time

        batch_times.append(time_taken)
        mean_time = sum(batch_times) / len(batch_times)

        est_remaining = (len(batches) - i - 2) * mean_time

        remaining_td = timedelta(seconds=round(est_remaining))
        time_td = timedelta(seconds=round(time_taken))
        mean_td = timedelta(seconds=round(mean_time))

        log.info(f"Batch {i} time: {time_td} (avg: {mean_td}, remaining: {remaining_td}")

    log.info(f"Processing batches complete. {apps_written} apps written")


def get_app_data(app_list: pd.DataFrame, start: int, stop: int, pause: int) -> list:
    """Return list of app data generated from parser."""
    app_data = []

    # iterate through each row of app_list, confined by start and stop
    for index, row in app_list[start:stop].iterrows():
        appid = row["appid"]
        name = row["name"]

        log.info(f"index: {index} || {appid} => {name}")

        # retrive app data for a row, handled by supplied parser, and append to list
        full_data_pl = parse_steam_request(appid=appid, name=name, params={"cc": "pl", "appids": appid})
        app_data.append(full_data_pl)
        price_eur = parse_steam_request(
            appid=appid,
            name=name,
            params={"cc": "de", "appids": appid, "filters": "price_overview"},
        )
        price_usd = parse_steam_request(
            appid=appid,
            name=name,
            params={"cc": "en", "appids": appid, "filters": "price_overview"},
        )

        log.warning(price_eur)
        log.warning(price_usd)

        price_eur = price_eur["price_overview"] if "price_overview" in price_eur else None
        price_usd = price_usd["price_overview"] if "price_overview" in price_usd else None

        full_data_pl["price_overview"] = (
            [full_data_pl["price_overview"], price_eur, price_usd] if "price_overview" in full_data_pl else None
        )

        time.sleep(pause)  # prevent overloading api with requests

    return app_data


def fetch_data(reset: bool = False) -> None:
    steam_spy_1000_json = get_request(url=STEAMSPY_URL, params={"request": "all"})

    apps_df = (
        pd.DataFrame.from_dict(steam_spy_1000_json, orient="index")[["appid", "name"]]
            .sort_values("appid")
            .reset_index()
    )

    if reset:
        reset_index(filepath=PATH_INDEX)

    index = get_index(filepath=PATH_INDEX)  # Retrieve last index downloaded from file

    prepare_data_file(filepath=PATH_CSV, index=index, columns=INTERESTING_DATA)  # Wipe or create file if index is 0

    process_batches(
        apps_df=apps_df,
        data_filepath=PATH_CSV,
        index_filepath=PATH_INDEX,
        columns=INTERESTING_DATA,
        begin=index,
        end=250,
        batch_size=5,
    )


def delete_document(doc: Document) -> None:
    log.info(f"Deleting doc {doc.id} => {doc.to_dict()}")
    doc.reference.delete()


def delete_all_documents(coll_ref: CollectionReference, batch_size: int) -> None:
    log.info(f'============================ DELETING DOCUMENTS IN BATCHES OF {batch_size} ============================')
    docs = coll_ref.limit(batch_size).stream()
    deleted = 0

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(delete_document, doc) for doc in docs]

        for _ in as_completed(futures):
            deleted += 1

    if deleted >= batch_size:
        return delete_all_documents(coll_ref, batch_size)
    return None


def remove_data(db: Client, batch_size: int = 10) -> None:
    for collection in db.collections():
        if collection.id in ["games"]:
            delete_all_documents(collection, batch_size)


def single_upload(db: Client, index: Hashable, row: Series) -> None:
    row = row.to_dict()
    release_date = row["release_date"]["date"]

    release_date_dict = {}
    if release_date:
        release_date_dict = {"release_date": datetime.strptime(release_date, "%d %b, %Y")}

    game_content = {
        "name": row["name"],
        "developers": row["developers"],
        "short_description": row["short_description"],
        **release_date_dict,
    }

    game = db.collection("games").document(str(row["steam_appid"]))

    game.set(game_content)
    log.debug(f'Game document {row["steam_appid"]} => {game_content} added.')

    if price_overview := row["price_overview"]:
        for price in price_overview:
            if not price:
                continue

            price_content = {
                "initial": price["initial"] / 100,
                "final": price["final"] / 100,
                "discount_percent": price["discount_percent"],
            }

            game.collection("prices").document(price["currency"]).set(price_content)
            log.debug(f'Price document {price["currency"]} => {price_content} added.')

    if screenshots := row["screenshots"]:
        for screenshot in screenshots:
            if not screenshot:
                continue

            screenshot_content = {
                "path_thumbnail": screenshot["path_thumbnail"],
                "path_full": screenshot["path_full"],
            }

            game.collection("screenshots").document(str(screenshot["id"])).set(screenshot_content)
            log.debug(f'Screenshot document {screenshot["id"]} => {screenshot_content} added.')

    log.info(f'{index} Document {row["steam_appid"]} => {row["name"]} processed successfully.')


def upload_data(db: Client) -> None:
    def as_literal(x: Any) -> Any:
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError) as exception:
            log.debug(f"'{exception}' was caught when parsing-> '{x}' type: '{type(x)}'")
            return None

    cols_to_get = [
        "name",
        "steam_appid",
        "short_description",
        "developers",
        "price_overview",
        "screenshots",
        "release_date",
    ]

    columns_to_convert = [
        "developers",
        "price_overview",
        "screenshots",
        "release_date",
    ]

    converters = {
        column: lambda x: as_literal(x) for column in columns_to_convert  # pylint: disable=unnecessary-lambda
    }

    data_df = pd.read_csv(PATH_CSV, usecols=cols_to_get, converters=converters)

    log.info('============================ UPLOADING DOCUMENTS STARTED ============================')

    with ThreadPoolExecutor() as executor:
        # pylint: disable=expression-not-assigned
        [executor.submit(single_upload, db, index, row) for index, row in data_df.iterrows()]

    log.info('============================ UPLOADING DOCUMENTS FINISHED ============================')


def main():
    start = perf_counter()

    cred = credentials.Certificate(PATH_CREDENTIALS)
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    # fetch_data(reset=False)
    remove_data(db=db, batch_size=10)
    upload_data(db=db)

    log.info(f"SCRIPT TOOK: {perf_counter() - start:.3f} s")


if __name__ == "__main__":
    main()
