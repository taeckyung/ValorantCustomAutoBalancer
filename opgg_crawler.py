
# %%
import json
import time
import requests
import pandas as pd

from typing import *
from pprint import pprint
from datetime import datetime, timedelta


# %%
MAX_SEARCH_PAGE_LIMIT = 100  # must be the multiple of 20
MAX_SEARCH_DAY_LIMIT = 30  # days
HEADER = {'Accept-Language':'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'}


user_df = pd.read_csv("account.csv", encoding="utf-8")
print(user_df)

match_candidates_to_search: Dict[str, Dict[str, str]] = dict()
user_stats_list: Dict[str, Dict] = dict()


# %%
# refresh records
for _, row in user_df.iterrows():
    name = row["gameName"].upper()
    tag = row["tagLine"].upper()

    user_name_merged = f"{name}#{tag}"
    if user_name_merged not in user_stats_list:
        user_stats_list[user_name_merged] = []

    url = f"https://valorant.op.gg/api/renew?gameName={name}&tagLine={tag}"
    requests.get(url,headers=HEADER)


# %%
# get match records
for _, row in user_df.iterrows():
    name = row["gameName"].upper()
    tag = row["tagLine"].upper()
    print(name, tag)

    for offset in range(0, MAX_SEARCH_PAGE_LIMIT+1, 20):
        url = f"https://valorant.op.gg/api/player/matches?gameName={name}&tagLine={tag}&queueId=custom&offset={offset}&limit=20"
        response = requests.get(url,headers=HEADER)
        match_record = json.loads(response.text)

        if (isinstance(match_record, dict)) and ('errorCode' in match_record.keys()):
            print(match_record)
            continue

        for match in match_record:
            if match['queueId'] != '':  # custom game does not have queueid
                continue

            if match["roundResults"] is None:  # invalid custom game
                continue

            if match["matchId"] in match_candidates_to_search:  # already exists
                continue

            round_result = str(match["roundResults"])
            win_count_l = int(round_result.split(":")[0])
            win_count_r = int(round_result.split(":")[1])

            if win_count_l != 13 and win_count_r != 13:  # properly ended games / exclude deathmatches
                continue

            match_time = datetime.strptime(match["gameStartDateTime"], "%Y-%m-%dT%H:%M:%S.%fZ")

            # only include within MAX_SEARCH_DAY_LIMIT days
            if match_time > datetime.now() - timedelta(days=MAX_SEARCH_DAY_LIMIT):  
                match_candidates_to_search[match['matchId']] = {
                    'name': name,
                    'tag': tag,
                    'gameStartDateTime': match_time
                }

# with open('data.json', 'w') as f:
#     json.dump(match_record_list, f)


# %%

for key in match_candidates_to_search.keys():
    match_id = key
    val = match_candidates_to_search[key]
    name = val['name'].upper()
    tag = val['tag'].upper()
    
    url = f"https://valorant.op.gg/api/player/matches/{match_id}?gameName={name}&tagLine={tag}"
    response = requests.get(url,headers=HEADER)
    match_record = json.loads(response.text)

    # validation
    if len(match_record["participants"]) != 10:
        continue

    for user in match_record["participants"]:
        if user["riotAccount"] is None:
            continue

        user_name = user["riotAccount"]["gameName"].upper()
        user_tag = user["riotAccount"]["tagLine"].upper()
        user_name_merged = f"{user_name}#{user_tag}"

        if user_name_merged not in user_stats_list:
            print(user_name_merged)
            continue

        user_stats_list[user_name_merged].append(
            {
                "win": float(user["won"]),
                "score": float(user["scorePerRound"]),
                "ranking": float(user["roundRanking"])
            }
        )


# %%
for user_name in user_stats_list.keys():
    df = pd.DataFrame.from_records(user_stats_list[user_name], columns=["win", "score", "ranking"])
    print("--------------------------------------")
    print(user_name)
    pprint(df.mean(axis=0))
    print("--------------------------------------")
