import time

from myShapley import Game
from utils import group_betweenness_digraph


def value_function_risk(s: set, ref_dict, nx_graph, mapping_dict):
    temporal_deviation = 0
    temp_time_start = time.time()
    for p in s:
        temporal_deviation += ref_dict[p]
    temp_time = time.time()-temp_time_start
    group = [mapping_dict[str(p)] for p in s]
    betweenness_time_start = time.time()
    betweenness = group_betweenness_digraph(nx_graph, group, normalized=False)
    betweenness_time = time.time()-betweenness_time_start
    return {"betweenness": betweenness, "temporal_deviation": temporal_deviation, "temp_duration": temp_time, "betweenness_duration": betweenness_time}

def create_risk_game(players: set):
    g = Game()
    g.players = players
    return g


