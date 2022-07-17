# for each timeline, read and parse it to snapshots of the game

from riot_api import ApiCaller
import asyncio

import time, re, json
import numpy as np
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


with open("data/matchIds.json") as fin:
    matchIds = json.load(fin)
with open("item.json") as fin:
    items = json.load(fin)
all_items = items["data"] # dict: {item_id: {name:, description:, colloq:, ....}, }



loop = asyncio.get_event_loop()
for id in matchIds:
    pass












