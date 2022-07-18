# for each timeline, read and parse it to snapshots of the game

from riot_api import ApiCaller
import asyncio

import time, re, os, json
import numpy as np
import pandas as pd
from tqdm import tqdm


def requestsLog(url, status, headers):
    print(url)
    print(status)
    print(headers)

region = "kr"
api_key = ""
panth = ApiCaller(region, api_key, errorHandling=True, requestsLoggingFunction=requestsLog, debug=True)


async def getTimeline(matchId):
    try:
        data = await panth.getTimeline(matchId)
        return data
    except Exception as e:
        print(e)

async def getMatchRawData(matchId):
    try:
        data = await panth.getMatchMeta(matchId)
        return data
    except Exception as e:
        print(e)


def init_champ_status(): # use a ndarray to record
    result = np.zeros((6, 56)) # 6 item slots, 56d features


with open("data/matchIds.json") as fin:
    matchIds = json.load(fin)
with open("item.json") as fin:
    items = json.load(fin)
all_items = items["data"] # dict: {item_id: {name:, description:, colloq:, ....}, }


champion_root = "../src/champ/en_nv/"
item_root = "../src/item/en/"

loop = asyncio.get_event_loop()
for id in matchIds:
    tttt = loop.run_until_complete(getTimeline(id)) # json file
    meta = tttt["meta"] # player, champion, position

    champs = [v[0] for _, v in meta.items()] # 10 participating champions
    profiles = [pd.read_csv(champion_root + ch.lower() + ".csv").values.tolist() for ch in champs] # for every champ, use a list to record its info

    status = [np.zeros(6, 56) for _ in range(10)] # for 10 champs, each has 6 item slots, feature dim is 56
    
    for frame in tttt["timeline"]["frames"]:
        pass

        


loop.close()










