import time

import constants
import storage
from myShapley import Game


def value_function_risk(s: set, ref_dict, nx_graph, mapping_dict):
    perspective1 = 0
    perspective1_start = time.time()
    for p in s:
        perspective1 += ref_dict[p]
    perspective1_duration = time.time()-perspective1_start
    perspective2_start = time.time()
    if constants.METHOD == "betweenness":
        # perspective2 = group_betweenness_digraph(nx_graph, group, normalized=False)
        perspective2 = sum([storage.betweenness_centralities[p] if p in storage.betweenness_centralities else 0 for p in s])
    elif constants.METHOD == "dominator":
        if len(s) > 0:
            perspective2 = sum([storage.count_domination[p] if p in storage.count_domination else 0 for p in s])
        else:
            perspective2 = 0
    else:
        raise ValueError("Unknown method")
    perspective2_duration = time.time()-perspective2_start
    return {"perspective1": perspective1, "perspective1_duration": perspective1_duration, "perspective2": perspective2, "perspective2_duration": perspective2_duration}

def create_risk_game(players: set):
    g = Game()
    g.players = players
    return g


