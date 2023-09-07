
# %%
import json
import time
import pickle
import random
import asyncio
import traceback
import aiohttp
import requests
import numpy as np
import pandas as pd
import dataframe_image as dfi

from typing import *
from pprint import pprint
from datetime import datetime, timedelta



################################################################################################
# Constant Variables
################################################################################################

HEADER = {'Accept-Language':'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'}
DEFAULT_MMR = 150
FRONT_CONST = 4
BACK_CONST = 1
EXPECTED_SCORE = 200
MAX_TCP_CONNECTIONS = 10

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


def _get_user_list(all=False) -> List[RiotID]:
    if all:
        user_df = pd.read_csv("member_all.csv", encoding="utf-8")
    else:
        user_df = pd.read_csv("member_current.csv", encoding="utf-8")
    return [RiotID(user[0], user[1]) for user in user_df.to_numpy().tolist()]


async def _refresh_match_records(user_list):
    conn = aiohttp.TCPConnector(limit=MAX_TCP_CONNECTIONS)
    async with aiohttp.ClientSession(connector=conn) as session:
        async def body(id):
            url = f"https://valorant.op.gg/api/renew?gameName={id.name}&tagLine={id.tag}"

            async with session.get(url, headers=HEADER) as r:
                # print(id, r.status)
                return r.status

        await asyncio.gather(*(body(user) for user in user_list))


async def _get_match_candidates(user_list, search_limit, descending=True) -> List[Tuple[datetime, str, RiotID]]:
    match_id_list = set()
    match_candidates_to_search = []
    
    conn = aiohttp.TCPConnector(limit=MAX_TCP_CONNECTIONS)
    async with aiohttp.ClientSession(connector=conn) as session:
        async def body(id, offset):
            url = f"https://valorant.op.gg/api/player/matches?gameName={id.name}&tagLine={id.tag}&queueId=custom&offset={offset}&limit=20"
            try:
                async with session.get(url, headers=HEADER) as r:
                    match_record = json.loads(await r.text())
            except Exception:
                traceback.format_exc()
                return

            if (isinstance(match_record, dict)) and ('errorCode' in match_record.keys()):
                return

            for match in match_record:
                if match['queueId'] != '':  # custom game does not have queueid
                    return

                if match["roundResults"] is None:  # invalid custom game
                    return

                if match["matchId"] in match_id_list:  # already exists
                    return

                round_result = str(match["roundResults"])
                win_count_l = int(round_result.split(":")[0])
                win_count_r = int(round_result.split(":")[1])

                if not ((win_count_l == 13 or win_count_r == 13) or (abs(win_count_l - win_count_r) == 2)):  # properly ended games / exclude deathmatches
                    return

                match_time = datetime.strptime(match["gameStartDateTime"].split("+")[0], "%Y-%m-%dT%H:%M:%S")

                # only include within MAX_SEARCH_DAY_LIMIT days
                # if match_time > datetime.now() - timedelta(days=day_limit):  
                match_id_list.add(match["matchId"])
                match_candidates_to_search.append((match_time, match['matchId'], id))

        await asyncio.gather(*(body(id, offset) for id in user_list for offset in range(0, search_limit+1, 20)))

    match_candidates_to_search.sort(reverse=True)

    return match_candidates_to_search


async def _get_match_records(match_candidates_to_search) -> List:
    conn = aiohttp.TCPConnector(limit=MAX_TCP_CONNECTIONS)
    async with aiohttp.ClientSession(connector=conn) as session:
        async def body(match_time, match_id, uid):
            url = f"https://valorant.op.gg/api/player/matches/{match_id}?gameName={uid.name}&tagLine={uid.tag}"

            try:
                async with session.get(url, headers=HEADER) as r:
                    match_record = json.loads(await r.text())
            except aiohttp.ClientConnectionError:
                traceback.format_exc()
                raise

            if (isinstance(match_record, dict)) and ('errorCode' in match_record.keys()):
                return

            # validation
            if len(match_record["participants"]) != 10:
                return

            return match_time, match_record

        result = await asyncio.gather(*(body(match_time, match_id, uid) for match_time, match_id, uid in match_candidates_to_search))
        result = [x for x in result if x is not None]  # remove None from exceptional cases
        result.sort(reverse=False)
        result = [x[1] for x in result]  # remove time and only maintain records
        return result


def _get_user_stats(user_list, match_record_list) -> pd.DataFrame:
    user_stats_list: Dict[str, List[Dict[str, float]]] = dict()

    for id in user_list:
        user_stats_list[str(id)] = []

    for match_record in match_record_list:
        # print(match_record)
        cnt = 0
        for user in match_record["participants"]:
            if user["gameName"] is None:
                continue

            user_id = RiotID(user["gameName"], user["tagLine"])
            if user_id.fullName not in user_stats_list:
                # print(user_id.fullName)
                continue

            cnt += 1

        if cnt < 5:
            continue

        for user in match_record["participants"]:
            if user["gameName"] is None:
                continue

            user_id = RiotID(user["gameName"], user["tagLine"])
            if user_id.fullName not in user_stats_list:
                # print(user_id.fullName)
                continue

            scorePerRound = user['score'] / user['rounds']
            # print(scorePerRound)

            user_stats_list[user_id.fullName].append(
                {
                    "win": float(user["won"]),
                    "score": float(scorePerRound),
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
        print("--------------------------------------")
        print(user_name)
        print(df)
        pprint(df.mean(axis=0))
        print("--------------------------------------")

    final_df["kda"] = final_df.apply(lambda r: (r["kills"] + r["assists"]) / r["deaths"], axis=1)
    mmr = _get_latest_mmr()
    final_df["mmr"] = mmr
    final_df = final_df.sort_values(by=["mmr"], ascending=False).round(2)

    return final_df


def _restricted_largest_differencing_method(array: List[Tuple[float, Any]]) -> List:
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


def _trans_mmr(MMR):
    if MMR<50:
        tMMR = 50
    elif MMR>300:
        tMMR = 300
    else:
        tMMR = 0.95*(MMR-150)+150
    # tMMR = 100/(1+np.exp(-(MMR-100)/50))+50
    return tMMR


def _update_mmr(match: Dict, mmr_list: pd.Series):
    # init
    elos, user_ids, fronts, backs = [], [], [], []

    match = match['participants']

    # check match
    if len(match) != 10:
        return mmr_list

    # check round
    rounds = match[0]['roundResults'].split(':')
    total_rounds = match[0]['rounds']
    win_rounds = np.abs(int(rounds[1]) - int(rounds[0]))

    # run by person
    for player in match:
        if_win = 1 if player['won'] else -1

        scorePerRound = player['score'] / player['rounds']

        front = FRONT_CONST * if_win * win_rounds
        back = BACK_CONST * total_rounds * (scorePerRound - EXPECTED_SCORE) / EXPECTED_SCORE
        # score = 1 / (FRONT_CONST + BACK_CONST) * (front + back)

        # save
        fronts.append(front)
        backs.append(back)

        # secret account
        if player['gameName'] == None:
            elos.append(DEFAULT_MMR * if_win)
            user_ids.append(None)
        else:
            id = RiotID(player['gameName'], player['tagLine']).fullName
            if id not in mmr_list.index:
                elos.append(DEFAULT_MMR * if_win)
                user_ids.append(None)
            else:
                elos.append(mmr_list[id] * if_win)
                user_ids.append(id)
    # make numpy
    elos = np.array(elos)
    user_ids = np.array(user_ids)

    # win rate version
    abs_elos = np.abs(elos)
    mean_elo = abs_elos.mean()
    winner_sum = abs_elos[elos >= 0].sum()
    losser_sum = abs_elos[elos <= 0].sum()
    rate_win = 2 * winner_sum / (winner_sum + losser_sum)
    rate_loss = 2 * losser_sum / (winner_sum + losser_sum)

    dMMR_front = np.array([1/rate_win*fronts[i] if elo_in>=0
                     else rate_loss*fronts[i] for i, elo_in in enumerate(elos)]) / 8
    dMMR_back = np.array([mean_elo/abs_elos[i] * back if back>=0
                          else abs_elos[i]/mean_elo * back for i, back in enumerate(backs)]) / 8
    dMMR = dMMR_front + dMMR_back

    # update
    for j, id in enumerate(user_ids):
        if id is None:
            continue
        mmr_list[id] += dMMR[j]
        mmr_list[id] = _trans_mmr(mmr_list[id])

    return mmr_list


def _get_latest_mmr():
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
        if not np.isnan(discord):
            content = content.replace(riot, f"<@{int(discord)}> ({riot})")

    return content    


async def get_member() -> str:
    user_list = [str(user) for user in _get_user_list()]
    return "현재 멤버: " + " / ".join(user_list)


async def add_member(members: str) -> str:
    result_str = ""

    for member in members.split(','):
        member = member.strip()
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

        all_user_df = pd.read_csv("member_all.csv", encoding="utf-8")

        if all_user_df["gameName"].isin([id.name]).any() and all_user_df["tagLine"].isin([id.tag]).any():
            continue

        all_user_df = pd.concat([all_user_df, pd.Series({'gameName': id.name, 'tagLine': id.tag, 'discordId': None}).to_frame().T], ignore_index=True)
        all_user_df.to_csv("member_all.csv", encoding="utf-8", index=False, header=True)

    return result_str


async def remove_member(members: str) -> str:
    result_str = ""

    for member in members.split(','):
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
    user_list = _get_user_list(all=True)

    try:
        with open(file="match_record.pickle", mode="rb") as f:
            data = pickle.load(f)
            last_updated = data["time"]
            match_record_list = data["match_record_list"][::-1][:]
    except:
        return "(오류) 데이터를 업데이트 해주세요.", None
    
    latest_match_time = datetime.strptime(match_record_list[0]['participants'][0]["gameStartDateTime"], 
                                            "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(hours=9)

    final_df = _get_user_stats(user_list, match_record_list)    
    print(final_df)
    dfi.export(final_df, 'table.png', max_cols=-1, max_rows=-1)

    # string += f"```{final_df.to_markdown(tablefmt='github', floatfmt='.1f')}```"
    return f"Latest Update: {last_updated.isoformat().replace('T', ' ')} \t|\t Latest Match: {latest_match_time.isoformat().replace('T', ' ')}\n", "table.png"


async def auto_balance() -> str:
    user_list = _get_user_list(all=False)

    if len(user_list) > 10:
        return "(오류) 멤버가 10명보다 많습니다. 멤버를 삭제해주세요."
    elif len(user_list) < 10:
        return "(오류) 멤버가 10명보다 적습니다. 멤버를 추가해주세요."

    with open(file="match_record.pickle", mode="rb") as f:
        data = pickle.load(f)
        last_updated = data["time"]
        
    df = _get_latest_mmr()
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

    team_l, team_r, sum_l, sum_r = _restricted_largest_differencing_method(rating_list)

    team_l = [x[1] for x in team_l]
    team_r = [x[1] for x in team_r]

    return f"Latest Update: {last_updated.isoformat().replace('T', ' ')}, Latest Match: {df.name} (최근 100경기)\nTeam A: {' / '.join(team_l)} (average rating: {sum_l/5:.2f})\nTeam B: {' / '.join(team_r)} (average rating: {sum_r/5:.2f})\n"


async def update() -> str:
    user_list = _get_user_list(all=True)
    await _refresh_match_records(user_list)
    match_candidates_to_search = await _get_match_candidates(user_list, 500, descending=False)
    match_record_list = await _get_match_records(match_candidates_to_search)

    if len(match_record_list) == 0:
        return "(오류) 매치 기록이 존재하지 않습니다."

    data = {
        "time": datetime.now(),
        "match_record_list": match_record_list
    }
    with open(file="match_record.pickle", mode="wb") as f:
        pickle.dump(data, f)

    user_list = [str(id) for id in user_list]
    time_list = []
    df_list = []
    mmr_list = pd.Series(data=np.ones(len(user_list)) * DEFAULT_MMR, index=user_list)

    for match in match_record_list:
        mmr_list = _update_mmr(match, mmr_list)
        df_list.append(mmr_list.copy())
        time = datetime.strptime(match['participants'][0]["gameStartDateTime"], "%Y-%m-%dT%H:%M:%S.%fZ") + timedelta(hours=9)
        print(time)
        time_list.append(time.isoformat().replace('T', ' '))
        
    df = pd.DataFrame(df_list, columns=user_list, index=time_list)
    df.to_csv(f"mmr.csv", encoding="utf-8", index=True, header=True)
    return f"매치 업데이트 (Latest match: {df.index[-1]})"


async def random_map() -> str():
    random.seed(time.time())
    map_list = ["스플릿", "로터스", "바인드", "프랙처", "헤이븐", "어센트", "아이스박스", "브리즈", "펄", "선셋"]
    random.shuffle(map_list)
    return ", ".join(map_list)


# %%
if __name__ == '__main__':
    print(asyncio.run(update()))
    # print(get_latest_mmr())
