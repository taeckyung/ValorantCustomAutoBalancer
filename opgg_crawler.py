
# %%
import asyncio
import json
import math
import time
import requests
import pandas as pd

from typing import *
from pprint import pprint
from datetime import datetime, timedelta


HEADER = {'Accept-Language':'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'}


async def get_stats(search_limit=100, day_limit=30):
    loop = asyncio.get_event_loop()

    match_candidates_to_search: Dict[str, Dict[str, str]] = dict()
    user_stats_list: Dict[str, Dict] = dict()
    latest_match_time = datetime.fromisocalendar(1900,1,1)
    total_match_cnt = 0

    user_df = pd.read_csv("account.csv", encoding="utf-8")

    future_list = []
    for _, row in user_df.iterrows():
        name = row["gameName"].upper()
        tag = row["tagLine"].upper()

        user_name_merged = f"{name}#{tag}"
        if user_name_merged not in user_stats_list:
            user_stats_list[user_name_merged] = []

        url = f"https://valorant.op.gg/api/renew?gameName={name}&tagLine={tag}"
        future_list.append(loop.run_in_executor(None, lambda: requests.get(url, headers=HEADER)))

    await asyncio.gather(*future_list)

    # get match records
    future_list = []
    for _, row in user_df.iterrows():
        name = row["gameName"].upper()
        tag = row["tagLine"].upper()
        # print(name, tag)

        for offset in range(0, search_limit+1, 20):
            url = f"https://valorant.op.gg/api/player/matches?gameName={name}&tagLine={tag}&queueId=custom&offset={offset}&limit=20"
            response = requests.get(url, headers=HEADER)
            match_record = json.loads(response.text)

            if (isinstance(match_record, dict)) and ('errorCode' in match_record.keys()):
                # print(match_record)
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
                if match_time > datetime.now() - timedelta(days=day_limit):  
                    match_candidates_to_search[match['matchId']] = {
                        'name': name,
                        'tag': tag,
                        'gameStartTime': match_time
                    }
    # with open('data.json', 'w') as f:
    #     json.dump(match_record_list, f)

    for key in match_candidates_to_search.keys():
        match_id = key
        val = match_candidates_to_search[key]
        name = val['name'].upper()
        tag = val['tag'].upper()
        
        url = f"https://valorant.op.gg/api/player/matches/{match_id}?gameName={name}&tagLine={tag}"
        response = requests.get(url, headers=HEADER)
        match_record = json.loads(response.text)

        if (isinstance(match_record, dict)) and ('errorCode' in match_record.keys()):
            print(match_record, flush=True)
            continue

        # validation
        if len(match_record["participants"]) != 10:
            continue

        total_match_cnt += 1
        match_time = val['gameStartTime']
        if latest_match_time < match_time:
            latest_match_time = match_time


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
                    "roundRanking": float(user["roundRanking"]),
                    "kills": float(user["kills"]),
                    "deaths": float(user["deaths"]),
                    "assists": float(user["assists"]),
                }
            )

    final_df = pd.DataFrame(columns=["win", "score", "roundRanking", "kills", "deaths", "assists"])
    for user_name in user_stats_list.keys():
        df = pd.DataFrame.from_records(user_stats_list[user_name], columns=["win", "score", "roundRanking", "kills", "deaths", "assists"])
        final_df.loc[user_name] = df.mean(axis=0)
    #     print("--------------------------------------")
    #     print(user_name)
    #     pprint(df.mean(axis=0))
    #     print("--------------------------------------")

    final_df["kda"] = final_df.apply(lambda r: (r["kills"] + r["assists"]) / r["deaths"], axis=1)
    final_df["rating"] = final_df.apply(lambda r: math.sqrt(r["win"]) * r["score"], axis=1)
    final_df = final_df.sort_values(by=["rating"], ascending=False)
    
    latest_match_time += timedelta(hours=9)  # UTC+9
    string = f"Total Matches: {total_match_cnt} \t|\t Latest Match: {latest_match_time.isoformat()} \t|\t rating := sqrt(win) * score\n"
    string += f"```{final_df.to_markdown()}```"
    return string


if __name__ == '__main__':
    print(asyncio.run(get_stats()))
# %%
