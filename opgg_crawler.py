
# %%
import asyncio
import json
import math
import requests
import numpy as np
import pandas as pd
import dataframe_image as dfi

from typing import *
from datetime import datetime, timedelta


HEADER = {'Accept-Language':'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'}


class RiotID:
    def __init__(self, name: str, tag: str = None) -> None:
        name = name.upper()
        if tag is None:
            assert(name.count("#") == 1)
            self.name, self.tag = name.split("#")
        else:
            tag = tag.upper()
            assert(name.count("#") == 0 and tag.count("#") == 0)
            self.name = name
            self.tag = tag
        self.fullName = f"{self.name}#{self.tag}"

    def __str__(self) -> str:
        return self.fullName


async def get_member() -> str:
    user_df = pd.read_csv("account.csv", encoding="utf-8")
    user_list = [str(RiotID(user[0], user[1])) for user in user_df.to_numpy().tolist()]
    return "현재 멤버: " + str(user_list)


async def add_member(members: str) -> str:
    result_str = ""

    for member in members.split():
        try:
            id = RiotID(member)
        except:
            result_str += f"(오류) 닉네임#태그 형태로 입력해주세요: {member}\n"
            continue

        user_df = pd.read_csv("account.csv", encoding="utf-8")

        if user_df["gameName"].isin([id.name]).any() and user_df["tagLine"].isin([id.tag]).any():
            result_str += f"(오류) 이미 존재하는 멤버입니다: {member}\n"
            continue

        user_df = pd.concat([user_df, pd.Series({'gameName': id.name, 'tagLine': id.tag}).to_frame().T], ignore_index=True)
        user_df.to_csv("account.csv", encoding="utf-8", index=False, header=True)
        result_str += f"멤버가 추가되었습니다: {member}\n"

    return result_str


async def remove_member(members: str) -> str:
    result_str = ""

    for member in members.split():
        try:
            id = RiotID(member)
        except:
            result_str += f"(오류) 닉네임#태그 형태로 입력해주세요: {member}\n"
            continue

        user_df = pd.read_csv("account.csv", encoding="utf-8")
        existance = user_df["gameName"].isin([id.name]) & user_df["tagLine"].isin([id.tag])
        if not existance.any():
            result_str += f"(오류) 존재하지 않는 멤버입니다: {member}\n"
            continue

        user_df = user_df.drop([existance.idxmax()])
        user_df.to_csv("account.csv", encoding="utf-8", index=False, header=True)
        result_str += f"멤버가 삭제되었습니다: {member}\n"

    return result_str

async def get_stats(search_limit=100, day_limit=30) -> Tuple[str, datetime, pd.DataFrame]:
    loop = asyncio.get_event_loop()

    match_candidates_to_search: Dict[str, Dict[str, str]] = dict()
    user_stats_list: Dict[str, Dict] = dict()
    latest_match_time = datetime.fromisocalendar(1900,1,1)
    total_match_cnt = 0

    user_df = pd.read_csv("account.csv", encoding="utf-8")

    future_list = []
    for _, row in user_df.iterrows():
        id = RiotID(row["gameName"], row["tagLine"])
        if str(id) not in user_stats_list:
            user_stats_list[str(id)] = []

        url = f"https://valorant.op.gg/api/renew?gameName={id.name}&tagLine={id.tag}"
        future_list.append(loop.run_in_executor(None, lambda: requests.get(url, headers=HEADER)))

    await asyncio.gather(*future_list)

    # get match records
    future_list = []
    for _, row in user_df.iterrows():
        id = RiotID(row["gameName"], row["tagLine"])

        for offset in range(0, search_limit+1, 20):
            url = f"https://valorant.op.gg/api/player/matches?gameName={id.name}&tagLine={id.tag}&queueId=custom&offset={offset}&limit=20"
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
                        'id': id,
                        'gameStartTime': match_time
                    }

    for key in match_candidates_to_search.keys():
        match_id = key
        val = match_candidates_to_search[key]
        id = val['id']
        
        url = f"https://valorant.op.gg/api/player/matches/{match_id}?gameName={id.name}&tagLine={id.tag}"
        response = requests.get(url, headers=HEADER)
        match_record = json.loads(response.text)

        if (isinstance(match_record, dict)) and ('errorCode' in match_record.keys()):
            # print(match_record, flush=True)
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

            user_id = RiotID(user["riotAccount"]["gameName"], user["riotAccount"]["tagLine"])
            if user_id.fullName not in user_stats_list:
                # print(user_id.fullName)
                continue

            user_stats_list[user_id.fullName].append(
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
    final_df = final_df.sort_values(by=["rating"], ascending=False).round(2)
    
    latest_match_time += timedelta(hours=9)  # UTC+9
    dfi.export(final_df, 'table.png', max_cols=-1, max_rows=-1)

    # string += f"```{final_df.to_markdown(tablefmt='github', floatfmt='.1f')}```"
    return total_match_cnt, latest_match_time, final_df


def restricted_largest_differencing_method(array: List[Tuple[float, Any]]) -> List:
    assert(len(array) % 2 == 0)
    array_sorted = sorted(array, reverse=True)
    diff_array = []
    for i in range(0, len(array_sorted), 2):
        diff_array.append(
            (array_sorted[i][0] - array_sorted[i+1][0], array_sorted[i], array_sorted[i+1])
        )
    diff_array = sorted(diff_array, reverse=True)

    set_l = []
    set_r = []

    for i in range(len(diff_array)):
        sum_l = np.sum([x[0] for x in set_l])
        sum_r = np.sum([x[0] for x in set_r])

        if sum_l < sum_r:
            set_l.append(diff_array[i][1])
            set_r.append(diff_array[i][2])
        else:
            set_l.append(diff_array[i][2])
            set_r.append(diff_array[i][1])

    return set_l, set_r, np.sum([x[0] for x in set_l]), np.sum([x[0] for x in set_r])


async def auto_balance() -> str:
    user_df = pd.read_csv("account.csv", encoding="utf-8")
    user_list = [RiotID(user[0], user[1]) for user in user_df.to_numpy().tolist()]

    if len(user_list) > 10:
        return "(오류) 멤버가 10명보다 많습니다. 멤버를 삭제해주세요."
    elif len(user_list) < 10:
        return "(오류) 멤버가 10명보다 적습니다. 멤버를 추가해주세요."
    
    _, _, df = await get_stats()
    df = df.loc[:, ["rating"]]
    rating_list = df.reset_index().to_numpy().tolist()
    rating_list = [(x[1], x[0]) for x in rating_list]

    # support for unknowns
    avg = np.nanmean([x[0] for x in rating_list])
    rating_list = [(avg, x[1]) if np.isnan(x[0]) else x for x in rating_list]

    team_l, team_r, sum_l, sum_r = restricted_largest_differencing_method(rating_list)

    team_l = [x[1] for x in team_l]
    team_r = [x[1] for x in team_r]

    return f"Team A: {team_l} (average rating: {sum_l/5:.2f})\nTeam B: {team_r} (average rating: {sum_r/5:.2f})\n"



# %%
if __name__ == '__main__':
    print(asyncio.run(get_stats()))
