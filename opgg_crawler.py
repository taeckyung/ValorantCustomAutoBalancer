
import json
import time
import requests
import pandas as pd

from typing import *
from pprint import pprint
from datetime import datetime, timedelta


MAX_SEARCH_LIMIT = 100  # must be the multiple of 20
HEADER = {'Accept-Language':'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'}


user_df = pd.read_csv("account.csv", encoding="utf-8")
print(user_df)

match_dict: Dict[str, Dict[str, str]] = dict()
user_record_df_dict: Dict[str, Dict] = dict()

for _, row in user_df.iterrows():
    name = row["gameName"].upper()
    tag = row["tagLine"].upper()
    print(name, tag)

    user_name_merged = f"{name}#{tag}"
    if user_name_merged not in user_record_df_dict:
        user_record_df_dict[user_name_merged] = []

    # refresh records
    url = f"https://valorant.op.gg/api/renew?gameName={name}&tagLine={tag}"
    requests.get(url,headers=HEADER)

    # time.sleep(1.0)

    for offset in range(0, MAX_SEARCH_LIMIT+1, 20):
        # match records
        url = f"https://valorant.op.gg/api/player/matches?gameName={name}&tagLine={tag}&queueId=custom&offset={offset}&limit=20"
        response = requests.get(url,headers=HEADER)
        match_record = json.loads(response.text)

        if (isinstance(match_record, dict)) and ('errorCode' in match_record.keys()):
            print(match_record)
            continue

        for match in match_record:
            if match['queueId'] == '':  # custom game does not have queueid
                if match["roundResults"] is not None:
                    round_result = str(match["roundResults"])
                    left = int(round_result.split(":")[0])
                    right = int(round_result.split(":")[1])
                    if left > 12 or right > 12:  # properly ended games
                        match_time = datetime.strptime(match["gameStartDateTime"], "%Y-%m-%dT%H:%M:%S.%fZ")
                        if match_time > datetime.now() - timedelta(days=30):  # only include within 30 days
                            match_dict[match['matchId']] = {
                                'name': name,
                                'tag': tag,
                                'gameStartDateTime': match_time
                            }




# with open('data.json', 'w') as f:
#     json.dump(match_record_list, f)


match_record_list = []

for key in match_dict.keys():
    match_id = key
    val = match_dict[key]
    name = val['name'].upper()
    tag = val['tag'].upper()
    
    url = f"https://valorant.op.gg/api/player/matches/{match_id}?gameName={name}&tagLine={tag}"
    response = requests.get(url,headers=HEADER)
    match_record = json.loads(response.text)

    match_record_list.append(match_record)

    for user in match_record["participants"]:
        if user["riotAccount"] is None:
            continue

        user_name = user["riotAccount"]["gameName"].upper()
        user_tag = user["riotAccount"]["tagLine"].upper()
        user_name_merged = f"{user_name}#{user_tag}"

        if user_name_merged not in user_record_df_dict:
            print(user_name_merged)
            continue

        user_record_df_dict[user_name_merged].append(
            {
                "win": float(user["won"]),
                "score": float(user["scorePerRound"]),
                "ranking": float(user["roundRanking"])
            }
        )


for user_name in user_record_df_dict.keys():
    df = pd.DataFrame.from_records(user_record_df_dict[user_name], columns=["win", "score", "ranking"])
    print("--------------------------------------")
    print(user_name)
    pprint(df.mean(axis=0))
    print("--------------------------------------")

# pprint(user_record_df_dict)