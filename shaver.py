import itertools
import logging
import os
import pickle
import random
import sys
import time

import networkx as nx
import numpy as np
import pm4py
import pydotplus
from matplotlib import pyplot as plt
from visualize_graph import colorize_graph
from preprocessing_BPIC11 import DB
import utils
from prepare_players import prepare_with_rules
from risk_assessment import create_successive_std, import_bpic11
from value_functions.risk import value_function_risk, create_risk_game

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
rootLogger = logging.getLogger()


def perm_generator(seq):
    seen = set()
    length = len(seq)
    while True:
        perm = tuple(random.sample(seq, length))
        if perm not in seen:
            seen.add(perm)
            yield perm


def calculate_permutation(player_list, permutation, game, ref_dict, marginal_contributions, shapley_values, weight,
                          nx_graph, mapping_dict):
    duration_temp = 0
    duration_betweenness = 0
    for j, player in enumerate(player_list):
        player_index = permutation.index(player)


        members = set(permutation[:player_index + 1])
        values = game.get_existing_coalition_value(members)
        if values:
            # print("Reusing calculated value for coalition")
            coalition_with_player_values = values
        else:
            coalition_with_player_values = value_function_risk(members, ref_dict, nx_graph, mapping_dict)
            duration_temp += coalition_with_player_values["temp_duration"]
            duration_betweenness += coalition_with_player_values["betweenness_duration"]
            game.add_coalition(members, coalition_with_player_values)

        members = set(permutation[:player_index])
        values = game.get_existing_coalition_value(members)
        coalition_without_player_values = None
        if values:
            coalition_without_player_values = values
        else:
            coalition_without_player_values = value_function_risk(members, ref_dict, nx_graph, mapping_dict)
            duration_temp += coalition_without_player_values["temp_duration"]
            duration_betweenness += coalition_without_player_values["betweenness_duration"]
            game.add_coalition(members, coalition_without_player_values)


        marg_cont = {
            "betweenness": coalition_with_player_values["betweenness"] - coalition_without_player_values[
                "betweenness"],
            "temporal_deviation": coalition_with_player_values["temporal_deviation"] -
                                  coalition_without_player_values["temporal_deviation"]
        }
        marginal_contributions.append(marg_cont)
        shapley_values[player].append(marg_cont)

    # https://math.stackexchange.com/questions/3633382/how-can-one-compute-the-shapley-value-using-monte-carlo
    marginal_betweenness_contributions_norm = utils.normalize_list(
        [x["betweenness"] for x in marginal_contributions])
    marginal_temporal_deviation_contributions_norm = utils.normalize_list(
        [x["temporal_deviation"] for x in marginal_contributions])
    player_shap_list = list(
        map(lambda x, y: (weight * x) + ((1 - weight) * y), marginal_betweenness_contributions_norm,
            marginal_temporal_deviation_contributions_norm))

    if len(player_shap_list) > 1:
        prev_shap = sum(player_shap_list[:-1]) / len(player_shap_list[:-1])
        curr_shap = sum(player_shap_list) / len(player_shap_list)
        epsilon = abs(curr_shap - prev_shap)
    else:
        epsilon = 100.0
    return epsilon, duration_temp, duration_betweenness


def calculate_shapley_value(game, ref_dict, nx_graph, mapping_dict, weight, e=0.1, max_sample_size=10000,
                            approximate=True, debug=False):
    player_list = list(game.players)
    shapley_values = dict()
    print(f"Number of players: {len(player_list)} -- {player_list}")
    print("Computing Shapley for each player")
    for player in player_list:
        shapley_values[player] = []
    marginal_contributions = []
    epsilon = 100.0
    i = 0
    if approximate:
        print("Sampling from permutations")
        rand_perms = perm_generator(player_list)
        permutations = [list(next(rand_perms)) for _ in range(max_sample_size)]
        while epsilon > e and i < len(permutations):
            i += 1
            print(f"Iteration {i}")
            t_start_iter = time.time()
            permutation = permutations[i]
            epsilon, duration_temp, duration_betweenness = calculate_permutation(player_list, permutation, game,
                                                                                 ref_dict, marginal_contributions,
                                                                                 shapley_values, weight, nx_graph,
                                                                                 mapping_dict)
            print(f"Took {time.time() - t_start_iter}s")
    else:
        permutations = list(itertools.permutations(player_list))
        overall_duration_temp = 0
        overall_duration_betweennes = 0
        for i, permutation in enumerate(permutations):
            print(f"Permutation: {i}/{len(permutations)}")
            epsilon, duration_temp, duration_betweenness = calculate_permutation(player_list, permutation, game,
                                                                                 ref_dict, marginal_contributions,
                                                                                 shapley_values,
                                                                                 weight, nx_graph, mapping_dict)
            overall_duration_temp += duration_temp
            overall_duration_betweennes += duration_betweenness
        print(f"Took {overall_duration_temp}s for temporal deviation calculation")
        print(f"Took {overall_duration_betweennes}s for betweenness calculation")

    # Normalize
    res_norm = dict()
    res = dict()
    all_temp_devs = []
    all_betweenness = []
    for player, v in shapley_values.items():
        for el in v:
            all_temp_devs.append(el["temporal_deviation"])
            all_betweenness.append(el["betweenness"])

    min_betweenness_val = min(all_betweenness)
    max_betweenness_val = max(all_betweenness)
    min_temp_devs_val = min(all_temp_devs)
    max_temp_devs_val = max(all_temp_devs)
    print(
        f"Betweenness\nMin: {min_betweenness_val}, Max: {max_betweenness_val}\nTemporal Deviation\nMin: {min_temp_devs_val}, Max: {max_temp_devs_val}")
    for player, v in shapley_values.items():
        marginal_temporal_deviation_contributions_norm = []
        marginal_betweenness_contributions_norm = []
        if min_temp_devs_val == max_temp_devs_val:
            marginal_temporal_deviation_contributions_norm = [0] * len(v)
        else:
            ts = [x["temporal_deviation"] for x in v]
            for x in ts:
                marginal_temporal_deviation_contributions_norm.append(
                    (x - min_temp_devs_val) / (max_temp_devs_val - min_temp_devs_val))

        if min_betweenness_val == max_betweenness_val:
            marginal_betweenness_contributions_norm = [0] * len(v)
        else:
            bs = [x["betweenness"] for x in v]
            for x in bs:
                marginal_betweenness_contributions_norm.append(
                    (x - min_betweenness_val) / (max_betweenness_val - min_betweenness_val))

        comb_norm = list(map(lambda x, y: (weight * x) + ((1 - weight) * y), marginal_betweenness_contributions_norm,
                             marginal_temporal_deviation_contributions_norm))

        comb = list(map(lambda x, y: (weight * x) + ((1 - weight) * y), [x["betweenness"] for x in v],
                        [x["temporal_deviation"] for x in v]))

        res_norm[player] = sum(comb_norm) / len(comb_norm)
        res[player] = sum(comb) / len(comb)
    return res, res_norm, i


class Result:
    def __init__(self, res, res_norm, mapping_dict, graph, duration) -> None:
        self.duration = duration
        self.mapping_dict = mapping_dict
        self.result_normalized = res_norm
        self.result = res
        self.graph = graph


if __name__ == '__main__':
    DATASET = "BPIC11"
    TITLE = "results_" + DATASET
    # if True: Monte Carlo permutation sampling is applied
    # else the number of activities is filtered by the ER-Miner
    APPROXIMATE = True
    # maximum iterations for Monte Carlo permutation sampling
    SAMPLE_SIZE = 10000
    # dependency threshold for heuristics miner
    DEP_THRESH = 0.6
    # epsilon value for convergence
    BASE_EPS = 0.00001

    filt = ['M14', 'M15']  # corpus uteri
    db_path = "data/bpic11/" + DATASET + "_" + "-".join(filt) + ".pkl"

    log = traces_with_timestamps = mapping_dict = None
    if os.path.isfile(db_path):
        with open(db_path, "rb") as f:
            db = pickle.load(f)
            log = db.log
            traces_with_timestamps = db.traces_with_timestamps
            mapping_dict = db.mapping_dict
            avg = db.avg
            std = db.std
            ext = db.ext
            mysum = db.mysum
            mymax = db.mymax
    else:
        log, traces_with_timestamps, mapping_dict = import_bpic11(filt)
        avg, std, ext, mysum, mymax = create_successive_std(traces_with_timestamps=traces_with_timestamps)
        db = DB(log, mapping_dict, traces_with_timestamps, avg, std, ext, mysum, mymax, DATASET)

        with open(db_path, 'wb') as handle:
            pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)


    heu_net = pm4py.discover_heuristics_net(log, dependency_threshold=DEP_THRESH)
    graph = pm4py.visualization.heuristics_net.visualizer.get_graph(heu_net=heu_net)

    nx_graph = nx.nx_pydot.from_pydot(graph)
    print("Built graph")

    uid_to_label = dict()
    for n in nx_graph.nodes(data=True):
        uid_to_label[n[0]] = " ".join(n[1]["label"].split(" ")[:-1])
    nx_graph_relabeled = nx.relabel_nodes(nx_graph, uid_to_label)
    plt.close()

    # -------------------------------------------------------------------------
    coalition_path = DATASET + "_current_coalitions.pkl"
    sublog_path = DATASET + "_" + "-".join(filt)
    coalition_filepath = os.path.join("data", "bpic11", sublog_path, coalition_path)

    if APPROXIMATE:
        players = mapping_dict.keys()
    else:
        players = prepare_with_rules(traces_with_timestamps, threshold_range=[(0.85, 0.86)])
        players = [str(x) for x in list(players)]

    print(players)
    g = create_risk_game(players)
    # 0 = full temp deviation, 1 = full betweenness
    # number_range = np.arange(0, 1.05, 0.05)
    # formatted_number_range = [round(num, 2) for num in number_range]
    formatted_number_range = [0.1, 0.5, 0.9]  # adjust weights here
    for w in formatted_number_range:
        if w > 0.0:
            eps = BASE_EPS / w
        else:
            eps = BASE_EPS
        if os.path.isfile(coalition_filepath):
            with open(coalition_filepath, "rb") as f:
                g.coalitions = pickle.load(f)

        start_time = time.time()
        r1, normalized_r1, avg_samples = calculate_shapley_value(g, ref_dict=std, nx_graph=nx_graph_relabeled,
                                                                 mapping_dict=mapping_dict, weight=w, e=eps,
                                                                 max_sample_size=SAMPLE_SIZE,
                                                                 approximate=APPROXIMATE)
        overall_time = time.time() - start_time
        print(f"Took {overall_time}s")

        with open(coalition_filepath, 'wb') as handle:
            pickle.dump(g.coalitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Sort result
        normalized_r1 = {k: v for k, v in sorted(normalized_r1.items(), key=lambda item: item[1])}
        for node, shapleys in normalized_r1.items():
            print(f"Node '{mapping_dict[str(node)]}' has an impact of {normalized_r1[node]}")

        print("Writing colored graph")
        colorize_graph(normalized_r1=normalized_r1, graph=graph, mapping_dict=mapping_dict)

        pydotplus.graphviz.Dot.write(graph,
                                     "results/plots/{0}_{1}_{2}_{3}_{4}_{5}_{6}_graph.png".format(DATASET,
                                                                                                  str(SAMPLE_SIZE),
                                                                                                  str(DEP_THRESH),
                                                                                                  str(w),
                                                                                                  str(avg_samples),
                                                                                                  BASE_EPS,
                                                                                                      "-".join(
                                                                                                          filt)),
                                     format="png")

        # gather mapping dict with result
        r = Result(r1, normalized_r1, mapping_dict, graph, overall_time)

        save_path = "data/bpic11/" + sublog_path + "/{0}_{1}_{2}_{3}_{4}_{5}_{6}.pkl".format(DATASET,
                                                                                             str(SAMPLE_SIZE),
                                                                                             str(DEP_THRESH),
                                                                                             str(w),
                                                                                             str(avg_samples),
                                                                                             BASE_EPS,
                                                                                                 "-".join(filt))

        with open(save_path, 'wb') as handle:
            pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)

