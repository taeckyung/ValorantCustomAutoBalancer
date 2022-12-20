
# %%
import json
import time
import pickle
import random
import asyncio
import requests
import numpy as np
import pandas as pd
import dataframe_image as dfi

from typing import *
from datetime import datetime, timedelta



################################################################################################
# Constant Variables
################################################################################################

HEADER = {'Accept-Language':'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'}
DEFAULT_MMR = 150
FRONT_CONST = 4
BACK_CONST = 1
EXPECTED_SCORE = 150

################################################################################################
# Helper Functions
################################################################################################

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

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, RiotID):
            return False
        return self.fullName == __o.fullName


def get_user_list(all=False) -> List[RiotID]:
    if all:
        user_df = pd.read_csv("member_all.csv", encoding="utf-8")
    else:
        user_df = pd.read_csv("member_current.csv", encoding="utf-8")
    return [RiotID(user[0], user[1]) for user in user_df.to_numpy().tolist()]


def refresh_match_records(user_list):
    for id in user_list:
        url = f"https://valorant.op.gg/api/renew?gameName={id.name}&tagLine={id.tag}"
        requests.get(url, headers=HEADER)


def get_match_candidates(user_list, search_limit, descending=True) -> List[Tuple[datetime, str, RiotID]]:
    match_id_list = set()
    match_candidates_to_search = []

    for id in user_list:
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

                if match["matchId"] in match_id_list:  # already exists
                    continue

                round_result = str(match["roundResults"])
                win_count_l = int(round_result.split(":")[0])
                win_count_r = int(round_result.split(":")[1])

                if win_count_l != 13 and win_count_r != 13:  # properly ended games / exclude deathmatches
                    continue

                match_time = datetime.strptime(match["gameStartDateTime"], "%Y-%m-%dT%H:%M:%S.%fZ")

                # only include within MAX_SEARCH_DAY_LIMIT days
                # if match_time > datetime.now() - timedelta(days=day_limit):  
                match_id_list.add(match["matchId"])
                match_candidates_to_search.append((match_time, match['matchId'], id))

    match_candidates_to_search.sort(reverse=descending)

    return match_candidates_to_search


def get_match_records(match_candidates_to_search) -> List:
    match_record_list = []

    for _, match_id, uid in match_candidates_to_search:
        url = f"https://valorant.op.gg/api/player/matches/{match_id}?gameName={uid.name}&tagLine={uid.tag}"
        response = requests.get(url, headers=HEADER)
        match_record = json.loads(response.text)

        if (isinstance(match_record, dict)) and ('errorCode' in match_record.keys()):
            # print(match_record, flush=True)
            continue

        # validation
        if len(match_record["participants"]) != 10:
            continue

        match_record_list.append(match_record)

    return match_record_list 


def get_user_stats(user_list, match_record_list) -> pd.DataFrame:
    user_stats_list: Dict[str, List[Dict[str, float]]] = dict()

    for id in user_list:
        user_stats_list[str(id)] = []

    for match_record in match_record_list:
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
                    "ranking": float(user["roundRanking"]),
                    "kills": float(user["kills"]),
                    "deaths": float(user["deaths"]),
                    "assists": float(user["assists"]),
                }
            )

    final_df = pd.DataFrame(columns=["win", "score", "ranking", "kills", "deaths", "assists"])
    for user_name in user_stats_list.keys():
        df = pd.DataFrame.from_records(user_stats_list[user_name], columns=["win", "score", "ranking", "kills", "deaths", "assists"])
        final_df.loc[user_name] = df.mean(axis=0)
    #     print("--------------------------------------")
    #     print(user_name)
    #     pprint(df.mean(axis=0))
    #     print("--------------------------------------")

    final_df["kda"] = final_df.apply(lambda r: (r["kills"] + r["assists"]) / r["deaths"], axis=1)
    mmr = get_latest_mmr()
    final_df["mmr"] = mmr
    final_df = final_df.sort_values(by=["mmr"], ascending=False).round(2)

    return final_df


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


def trans_mmr(MMR):
    if MMR<50:
        tMMR = 50
    elif MMR>300:
        tMMR = 300
    else:
        tMMR = 0.97*(MMR-150)+150
    # tMMR = 100/(1+np.exp(-(MMR-100)/50))+50
    return tMMR


def update_mmr(match, mmr_list, name_to_id):
    # init
    bef_MMR = mmr_list.copy()
    elos, idxs, fronts, backs = [], [], [], []
    cnt_games = np.zeros((2, len(mmr_list)))

    match = match['participants']

    # check match
    if len(match) != 10:
        return mmr_list, cnt_games, np.zeros(10)

    # check round
    rounds = match[0]['roundResults'].split(':')
    total_rounds = match[0]['rounds']
    if int(rounds[0]) > int(rounds[1]):
        won_rounds = int(rounds[0]) - int(rounds[1])
    else:
        won_rounds = int(rounds[1]) - int(rounds[0])

    # run by person
    for i in range(len(match)):
        now_people = match[i]
        won_value = 1 if now_people['won'] else -1

        front = FRONT_CONST * won_value * won_rounds
        back = BACK_CONST * (
                    total_rounds * now_people['scorePerRound'] - total_rounds * EXPECTED_SCORE) / EXPECTED_SCORE
        # score = 1 / (FRONT_CONST + BACK_CONST) * (front + back)

        # save
        fronts.append(front)
        backs.append(back)

        # secret account
        if now_people['riotAccount'] == None:
            elos.append(DEFAULT_MMR * won_value)
            idxs.append(-1)
        else:
            name = RiotID(now_people['riotAccount']['gameName'], now_people['riotAccount']['tagLine']).fullName
            if name not in name_to_id:
                elos.append(DEFAULT_MMR * won_value)
                idxs.append(-1)
            else:
                idx = name_to_id[name]
                elos.append(bef_MMR[idx] * won_value)
                idxs.append(idx)
                if won_value == 1:
                    cnt_games[0][idx] += 1
                else:
                    cnt_games[1][idx] += 1

    # make numpy
    elos = np.array(elos)
    idxs = np.array(idxs)

    # win rate version
    abs_elos = np.abs(elos)
    mean_elo = abs_elos.mean()
    winner_sum = elos[elos>=0].sum()
    losser_sum = abs_elos[elos<=0].sum()
    rate_win = 2 * winner_sum / (winner_sum + losser_sum)
    rate_loss = 2 * losser_sum / (winner_sum + losser_sum)

    dMMR = np.array([1/rate_win*fronts[i]+mean_elo/elo_in*backs[i] if elo_in>=0
                     else rate_loss*fronts[i]+mean_elo/elo_in*backs[i] for i, elo_in in enumerate(abs_elos)])/8

    # update
    for j, id in enumerate(idxs):
        if id == -1:
            continue
        mmr_list[id] += dMMR[j]
        mmr_list[id] = trans_mmr(mmr_list[id])

    return mmr_list


def get_latest_mmr():
    try:
        df = pd.read_csv("mmr.csv", index_col=0)
        return df.iloc[-1].T
    except:
        return None


################################################################################################
# Discord API
################################################################################################

async def riotID_to_discord(content: str) -> str:
    df = pd.read_csv("member_all.csv", encoding="utf-8")
    user_list = [(str(RiotID(user[0], user[1])), user[2]) for user in df.to_numpy().tolist()]

    for riot, discord in user_list:
        content = content.replace(riot, f"<@{discord}>")

    return content    


async def get_member() -> str:
    user_list = [str(user) for user in get_user_list()]
    return "현재 멤버: " + " / ".join(user_list)


async def add_member(members: str) -> str:
    result_str = ""

    for member in members.split():
        try:
            id = RiotID(member)
        except:
            result_str += f"(오류) 닉네임#태그 형태로 입력해주세요: {member}\n"
            continue

        user_df = pd.read_csv("member_current.csv", encoding="utf-8")

        if user_df["gameName"].isin([id.name]).any() and user_df["tagLine"].isin([id.tag]).any():
            result_str += f"(오류) 이미 존재하는 멤버입니다: {member}\n"
            continue

        user_df = pd.concat([user_df, pd.Series({'gameName': id.name, 'tagLine': id.tag}).to_frame().T], ignore_index=True)
        user_df.to_csv("member_current.csv", encoding="utf-8", index=False, header=True)
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

        user_df = pd.read_csv("member_current.csv", encoding="utf-8")
        existance = user_df["gameName"].isin([id.name]) & user_df["tagLine"].isin([id.tag])
        if not existance.any():
            result_str += f"(오류) 존재하지 않는 멤버입니다: {member}\n"
            continue

        user_df = user_df.drop([existance.idxmax()])
        user_df.to_csv("member_current.csv", encoding="utf-8", index=False, header=True)
        result_str += f"멤버가 삭제되었습니다: {member}\n"

    return result_str


async def get_stats() -> str:
    user_list = get_user_list(all=True)

    try:
        with open(file="match_record.pickle", mode="rb") as f:
            data = pickle.load(f)
            last_updated = data["time"]
            match_record_list = data["match_record_list"][::-1][:30]
    except:
        return "(오류) 데이터를 업데이트 해주세요."
    
    latest_match_time = datetime.strptime(match_record_list[0]['participants'][0]["gameStartDateTime"], 
                                            "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(hours=9)

    final_df = get_user_stats(user_list, match_record_list)    
    dfi.export(final_df, 'table.png', max_cols=-1, max_rows=-1)

    # string += f"```{final_df.to_markdown(tablefmt='github', floatfmt='.1f')}```"
    return f"Latest Update: {last_updated.isoformat().replace('T', ' ')} \t|\t Latest Match: {latest_match_time.isoformat().replace('T', ' ')}\n", "table.png"


async def auto_balance() -> str:
    user_list = get_user_list(all=False)

    if len(user_list) > 10:
        return "(오류) 멤버가 10명보다 많습니다. 멤버를 삭제해주세요."
    elif len(user_list) < 10:
        return "(오류) 멤버가 10명보다 적습니다. 멤버를 추가해주세요."

    with open(file="match_record.pickle", mode="rb") as f:
        data = pickle.load(f)
        last_updated = data["time"]
        
    df = get_latest_mmr()
    if df is None:
        return "(오류) 데이터를 업데이트 해주세요."
        
    rating_list = []
    for user in user_list:
        user = str(user)
        if user in df.index.to_list():
            rating_list.append((df[user], user))
        else:
            rating_list.append((DEFAULT_MMR, user))

    # support for unknowns
    avg = np.nanmean([x[0] for x in rating_list])
    rating_list = [(avg, x[1]) if np.isnan(x[0]) else x for x in rating_list]

    team_l, team_r, sum_l, sum_r = restricted_largest_differencing_method(rating_list)

    team_l = [x[1] for x in team_l]
    team_r = [x[1] for x in team_r]

    return f"Latest Update: {last_updated.isoformat().replace('T', ' ')}, Latest Match: {df.name} (최근 30경기)\nTeam A: {' / '.join(team_l)} (average rating: {sum_l/5:.2f})\nTeam B: {' / '.join(team_r)} (average rating: {sum_r/5:.2f})\n"


async def update() -> str:
    user_list = get_user_list(all=True)
    refresh_match_records(user_list)
    match_candidates_to_search = get_match_candidates(user_list, 500, descending=False)
    match_record_list = get_match_records(match_candidates_to_search)

    if len(match_record_list) == 0:
        return "(오류) 매치 기록이 존재하지 않습니다."

    data = {
        "time": datetime.now(),
        "match_record_list": match_record_list
    }
    with open(file="match_record.pickle", mode="wb") as f:
        pickle.dump(data, f)

    time_list = []
    df_list = []
    mmr_list = np.ones(len(user_list)) * DEFAULT_MMR
    name_to_id = {user_list[i].fullName: i for i in range(len(user_list))}

    for match in match_record_list:
        mmr_list = update_mmr(match, mmr_list, name_to_id)
        df_list.append(mmr_list.copy())
        time = datetime.strptime(match['participants'][0]["gameStartDateTime"], "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(hours=9)
        time_list.append(time.isoformat().replace('T', ' '))
        
    df = pd.DataFrame(df_list, columns=user_list, index=time_list)
    df.to_csv(f"mmr.csv", encoding="utf-8", index=True, header=True)
    return f"매치 업데이트 (Latest match: {df.index[-1]})"


async def random_map() -> str():
    random.seed(time.time())
    map_list = ["바인드", "프랙처", "헤이븐", "어센트", "아이스박스", "브리즈", "펄"]
    random.shuffle(map_list)
    return ", ".join(map_list)


# %%
if __name__ == '__main__':
    print(asyncio.run(random_map()))
    # print(get_latest_mmr())
