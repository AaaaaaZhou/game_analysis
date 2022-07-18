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
    # for every champ, use an ndarray to record its info
    # drop first 2 columns: idx, ability_id
    profiles = [pd.read_csv(champion_root + ch.lower() + ".csv").values[:, 2:] for ch in champs] 

    status = [np.zeros(6, 56) for _ in range(10)] # for all 10 champs, each has 6 item slots, feature dim is 56
    record_itemslot = {idx: [0, 0, 0, 0, 0, 0] for idx in range(1, 11)}

    # participantsId: 1 ~ 10
    for idx, frame in enumerate(tttt["timeline"]["info"]["frames"]):
        # frame: dict_keys(['events', 'participantFrames', 'timestamp'])
        for event in frame["events"]:
            if event["type"] == "ITEM_PURCHASED":
                itemId = event["itemId"]
                participantId = event["participantId"]

                item_info = all_items[itemId]
                name = item_info["name"]
                
                path = item_root + name.lower() + ".csv"
                if not os.path.exists(path):
                    continue
                item_val = pd.read_csv(path).values[0, 2:]

                if "from" in item_info.keys():
                    for ff in item_info["from"]:
                        if ff in record_itemslot[participantId]:
                            iiii = record_itemslot[participantId].index(ff)
                            status[participantId - 1][iiii] = 0
                            record_itemslot[participantId] = 0
                iiii = record_itemslot[participantId].index(0)
                status[participantId - 1][iiii] = item_val
                record_itemslot[participantId][iiii] = itemId

            elif event["type"] == "ITEM_UNDO":
                participantId = event["participantId"]
                beforeId = event["beforeId"]
                afterId = event["afterId"]

                if beforeId in record_itemslot[participantId]:
                    iiii = record_itemslot[participantId].index(beforeId)
                    status[participantId - 1][iiii] = 0
                    record_itemslot[participantId][iiii] = 0
                if afterId == 0:
                    continue

                item_info = all_items[afterId]
                name = item_info["name"]

                path = item_root + name.lower() + ".csv"
                if not os.path.exists(path):
                    continue
                item_val = pd.read_csv(path).values[0, 2:]

                iiii = record_itemslot[participantId].index(0)
                status[participantId - 1][iiii] = item_val
                record_itemslot[participantId][iiii] = afterId
            
            elif event["type"] == "ITEM_DESTROYED" or "ITEM_SOLD":
                pass
            elif event["type"] == "SKILL_LEVEL_UP":
                pass

        


loop.close()










