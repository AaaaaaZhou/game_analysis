# rom pantheon import pantheon
from riot_api import ApiCaller
import asyncio

import time, re, json
from tqdm import tqdm

region = "kr"
api_key = "RGAPI-e3bf5808-3d42-4605-b4be-7ef2d2a4d962"

seed_id = ["duimianxiaodai", "fragiIe", "0o0OoO", "Happiness21", "adasdasdfaf", "doujianghuimian", "2639450511967776"]


def requestsLog(url, status, headers):
    print(url)
    print(status)
    print(headers)


panth = ApiCaller(region, api_key, errorHandling=True, requestsLoggingFunction=requestsLog, debug=True)



async def getSummonerId(name):
    try:
        data = await panth.getSummonerByName(name)
        return data['puuid']
    except Exception as e:
        print(e)

async def getRecentMatchlist(accountId):
    try:
        data = await panth.getMatchList(accountId, params={"start": 0, "endIndex":10})
        return data
    except Exception as e:
        print(e)

async def getMatchRawData(matchId):
    try:
        data = await panth.getMatchMeta(matchId)
        return data
    except Exception as e:
        print(e)

'''
# rewrite this with new api
async def getRecentMatchlist(accountId):
    try:
        data = await panth.getMatchlist(accountId, params={"endIndex":10})
        return data
    except Exception as e:
        print(e)


async def getRecentMatches(accountId):
    try:
        matchlist = await getRecentMatchlist(accountId)
        tasks = [panth.getMatch(match['gameId']) for match in matchlist['matches']]
        return await asyncio.gather(*tasks)
    except Exception as e:
        print(e)
'''
def saveFiles(match, visited, seed):
    with open("data/matchIds.json", "w") as fout:
        json.dump(list(match), fout)
    with open("data/visited_id.json", "w") as fout:
        json.dump(list(visited), fout)
    with open("data/seeds.json", "w") as fout:
        json.dump(seed, fout)



puuid_seed = [asyncio.run(getSummonerId(name)) for name in seed_id]
matches = set()
visited = set()
ts = time.time()
while puuid_seed and len(matches) < 10000:
    try:
        pl = puuid_seed.pop(0)
        recent = asyncio.run(getRecentMatchlist(pl))
        matches.update(recent)
        visited.add(pl)
        for m in recent:
            participants = asyncio.run(getMatchRawData(m))["metadata"]["participants"]
            for p in participants:
                if p not in visited and p not in puuid_seed:
                    puuid_seed.append(p)
        saveFiles(matches, visited, puuid_seed)
    except Exception as e:
        print(e)
        continue

te = time.time() - ts
hh = te // 3600
mm = (te % 3600) // 60
ss = te % 60

print("Execution time: %d hours  %d minutes  %d seconds" % (hh, mm, ss))

# summonerId, accountId, puu = asyncio.run(getSummonerId(name))

# print("s : ", summonerId)
# print("a : ", accountId)
# print("p : ", puu) # this works.
# recent = asyncio.run(getRecentMatchlist(puu))
# print(recent)
# partic = asyncio.run(getMatchRawData(recent[1]))["metadata"]["participants"]
# print(partic)











