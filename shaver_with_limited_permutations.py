import csv
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

import constants
import storage
from visualize_graph import colorize_graph
from preprocessing_BPIC11 import DB
import utils
from prepare_players import prepare_with_rules
from risk_assessment import create_successive_std, import_bpic11, import_synthetic, import_bpic20_ID
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
    duration1 = 0
    duration2 = 0
    for j, player in enumerate(player_list):
        player_index = permutation.index(player)

        members = set(permutation[:player_index + 1])
        values = game.get_existing_coalition_value(members)
        coalition_with_player_values = None
        if values:
            coalition_with_player_values = values
        else:
            coalition_with_player_values = value_function_risk(members, ref_dict, nx_graph, mapping_dict)
            duration1 += coalition_with_player_values["perspective1_duration"]
            duration2 += coalition_with_player_values["perspective2_duration"]
            game.add_coalition(members, coalition_with_player_values)

        members = set(permutation[:player_index])
        values = game.get_existing_coalition_value(members)
        coalition_without_player_values = None
        if values:
            coalition_without_player_values = values
        else:
            coalition_without_player_values = value_function_risk(members, ref_dict, nx_graph, mapping_dict)
            duration1 += coalition_without_player_values["perspective1_duration"]
            duration2 += coalition_without_player_values["perspective2_duration"]
            game.add_coalition(members, coalition_without_player_values)

        marg_cont = {
            "perspective1": coalition_with_player_values["perspective1"] -
                            coalition_without_player_values["perspective1"],
            "perspective2": coalition_with_player_values["perspective2"] - coalition_without_player_values[
                "perspective2"]
        }
        marginal_contributions.append(marg_cont)
        shapley_values[player].append(marg_cont)

    marginal_perspective2_contributions_norm = utils.normalize_list(
        [x["perspective2"] for x in marginal_contributions])
    marginal_perspective1_deviation_contributions_norm = utils.normalize_list(
        [x["perspective1"] for x in marginal_contributions])
    player_shap_list = list(
        map(lambda x, y: (weight * x) + ((1 - weight) * y), marginal_perspective2_contributions_norm,
            marginal_perspective1_deviation_contributions_norm))

    if len(player_shap_list) > 1:
        prev_shap = sum(player_shap_list[:-1]) / len(player_shap_list[:-1])
        curr_shap = sum(player_shap_list) / len(player_shap_list)
        epsilon = abs(curr_shap - prev_shap)
    else:
        epsilon = 100.0
    return epsilon, duration1, duration2


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
            if i % 1000 == 0:
                print(f"Currently at {i} samples")
            if i > max_sample_size - 1:
                raise IndexError("Exceeded maximum sample size")
            permutation = permutations[i]
            epsilon, duration_perspective1, duration_perspective2 = calculate_permutation(player_list, permutation,
                                                                                          game,
                                                                                          ref_dict,
                                                                                          marginal_contributions,
                                                                                          shapley_values, weight,
                                                                                          nx_graph,
                                                                                          mapping_dict)
        print(f"{i} samples")
    else:
        permutations = list(itertools.permutations(player_list))
        overall_duration_perspective1 = 0
        overall_duration_perspective2 = 0
        for i, permutation in enumerate(permutations):
            print(f"Permutation: {i}/{len(permutations)}")
            epsilon, duration_perspective1, duration_perspective2 = calculate_permutation(player_list, permutation,
                                                                                          game,
                                                                                          ref_dict,
                                                                                          marginal_contributions,
                                                                                          shapley_values,
                                                                                          weight, nx_graph,
                                                                                          mapping_dict)
            overall_duration_perspective1 += duration_perspective1
            overall_duration_perspective2 += duration_perspective2
        print(f"Took {overall_duration_perspective1}s for calculation of perspective 1")
        print(f"Took {overall_duration_perspective2}s for calculation of perspective 2")
    # Normalize
    res_norm = dict()
    res = dict()
    all_perspective1 = []
    all_perspective2 = []
    for player, v in shapley_values.items():
        for el in v:
            all_perspective1.append(el["perspective1"])
            all_perspective2.append(el["perspective2"])
    # sns.distplot(all_temp_devs, hist=True, kde=True,
    #              bins=int(180 / 5), color='darkblue',
    #              hist_kws={'edgecolor': 'black'},
    #              kde_kws={'linewidth': 4})
    # plt.show()
    min_perspective2_val = min(all_perspective2)
    max_perspective2_val = max(all_perspective2)
    min_perspective1_val = min(all_perspective1)
    max_perspective1_val = max(all_perspective1)
    for player, v in shapley_values.items():
        marginal_perspective1_contributions_norm = []
        marginal_perspective2_contributions_norm = []
        if min_perspective1_val == max_perspective1_val:
            marginal_perspective1_contributions_norm = [0] * len(v)
        else:
            ts = [x["perspective1"] for x in v]
            for x in ts:
                marginal_perspective1_contributions_norm.append(
                    (x - min_perspective1_val) / (max_perspective1_val - min_perspective1_val))

        if min_perspective2_val == max_perspective2_val:
            marginal_perspective2_contributions_norm = [0] * len(v)
        else:
            bs = [x["perspective2"] for x in v]
            for x in bs:
                marginal_perspective2_contributions_norm.append(
                    (x - min_perspective2_val) / (max_perspective2_val - min_perspective2_val))

        comb_norm = list(map(lambda x, y: (weight * x) + ((1 - weight) * y), marginal_perspective2_contributions_norm,
                             marginal_perspective1_contributions_norm))

        comb = list(map(lambda x, y: (weight * x) + ((1 - weight) * y), [x["perspective2"] for x in v],
                        [x["perspective1"] for x in v]))

        res_norm[player] = sum(comb_norm) / len(comb_norm)
        res[player] = sum(comb) / len(comb)
    return res, res_norm, i

def calculate_shapley_value_with_limited_permutations(game, control_flow_variants, ref_dict, nx_graph, mapping_dict, weight, e=0.1, max_sample_size=10000,
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
    # get the possible permutation list:
    permutations = []
    for k, v in control_flow_variants.items():
        # k is a set
        one_permutation = []
        for ele in k:
            activity_id = list(mapping_dict.keys())[list(mapping_dict.values()).index(ele)]  # Prints george
            one_permutation.append(activity_id)
        permutations.append(one_permutation)

    if approximate:
        print("Sampling from permutations")
        # rand_perms = perm_generator(player_list)
        # permutations = [list(next(rand_perms)) for _ in range(max_sample_size)]
        while epsilon > e and i < len(permutations)-1:
            i += 1
            if i % 1000 == 0:
                print(f"Currently at {i} samples")
            if i > max_sample_size - 1:
                raise IndexError("Exceeded maximum sample size")
            permutation = permutations[i]
            player_list = permutation.copy()
            player_list.sort()
            epsilon, duration_perspective1, duration_perspective2 = calculate_permutation(player_list, permutation,
                                                                                          game,
                                                                                          ref_dict,
                                                                                          marginal_contributions,
                                                                                          shapley_values, weight,
                                                                                          nx_graph,
                                                                                          mapping_dict)
        print(f"{i} samples")
    else:
        # permutations = list(itertools.permutations(player_list))
        overall_duration_perspective1 = 0
        overall_duration_perspective2 = 0
        for i, permutation in enumerate(permutations):
            print(f"Permutation: {i}/{len(permutations)}")
            player_list = permutation.copy()
            player_list.sort()
            epsilon, duration_perspective1, duration_perspective2 = calculate_permutation(player_list, permutation,
                                                                                          game,
                                                                                          ref_dict,
                                                                                          marginal_contributions,
                                                                                          shapley_values,
                                                                                          weight, nx_graph,
                                                                                          mapping_dict)
            overall_duration_perspective1 += duration_perspective1
            overall_duration_perspective2 += duration_perspective2
        print(f"Took {overall_duration_perspective1}s for calculation of perspective 1")
        print(f"Took {overall_duration_perspective2}s for calculation of perspective 2")
    # Normalize
    res_norm = dict()
    res = dict()
    all_perspective1 = []
    all_perspective2 = []
    for player, v in shapley_values.items():
        for el in v:
            all_perspective1.append(el["perspective1"])
            all_perspective2.append(el["perspective2"])
    # sns.distplot(all_temp_devs, hist=True, kde=True,
    #              bins=int(180 / 5), color='darkblue',
    #              hist_kws={'edgecolor': 'black'},
    #              kde_kws={'linewidth': 4})
    # plt.show()
    min_perspective2_val = min(all_perspective2)
    max_perspective2_val = max(all_perspective2)
    min_perspective1_val = min(all_perspective1)
    max_perspective1_val = max(all_perspective1)
    for player, v in shapley_values.items():
        marginal_perspective1_contributions_norm = []
        marginal_perspective2_contributions_norm = []
        if min_perspective1_val == max_perspective1_val:
            marginal_perspective1_contributions_norm = [0] * len(v)
        else:
            ts = [x["perspective1"] for x in v]
            for x in ts:
                marginal_perspective1_contributions_norm.append(
                    (x - min_perspective1_val) / (max_perspective1_val - min_perspective1_val))

        if min_perspective2_val == max_perspective2_val:
            marginal_perspective2_contributions_norm = [0] * len(v)
        else:
            bs = [x["perspective2"] for x in v]
            for x in bs:
                marginal_perspective2_contributions_norm.append(
                    (x - min_perspective2_val) / (max_perspective2_val - min_perspective2_val))

        comb_norm = list(map(lambda x, y: (weight * x) + ((1 - weight) * y), marginal_perspective2_contributions_norm,
                             marginal_perspective1_contributions_norm))

        comb = list(map(lambda x, y: (weight * x) + ((1 - weight) * y), [x["perspective2"] for x in v],
                        [x["perspective1"] for x in v]))
        if len(comb_norm) == 0:
            res_norm[player] = 0
        else:
            res_norm[player] = sum(comb_norm) / len(comb_norm)

        if len(comb) == 0:
            res[player] = 0
        else:
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
    log = traces_with_timestamps = mapping_dict = None
    if os.path.isfile(constants.DB_PATH):
        with open(constants.DB_PATH, "rb") as f:
            db = pickle.load(f) # db is an instance from class preprocessing_BPIC11.DB, it is a class with dictionaries /dataframes as attributes
            log = db.log # an instance of dataframe
            traces_with_timestamps = db.traces_with_timestamps
            mapping_dict = db.mapping_dict
            avg = db.avg
            med = db.med
            std = db.std
            ext = db.ext
            mysum = db.mysum
            mymax = db.mymax
    else:
        # log, traces_with_timestamps, mapping_dict = import_synthetic()
        log, traces_with_timestamps, mapping_dict = import_bpic20_ID()
        avg, med, std, ext, mysum, mymax = create_successive_std(traces_with_timestamps=traces_with_timestamps)
        db = DB(log, mapping_dict, traces_with_timestamps, avg, med, std, ext, mysum, mymax, constants.DATASET)

        with open(constants.DB_PATH, 'wb') as handle:
            pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)
    variants = pm4py.get_variants(log)
    heu_net = pm4py.discover_heuristics_net(log, dependency_threshold=constants.DEP_THRESH)
    graph = pm4py.visualization.heuristics_net.visualizer.get_graph(heu_net=heu_net)
    pm4py.view_heuristics_net(heu_net=heu_net)
    nx_graph = nx.nx_pydot.from_pydot(graph)
    print("Built graph")

    uid_to_label = dict()
    for n in nx_graph.nodes(data=True):
        uid_to_label[n[0]] = " ".join(n[1]["label"].split(" ")[:-1])
    nx_graph_relabeled = nx.relabel_nodes(nx_graph, uid_to_label)
    plt.close()

    if constants.METHOD == "dominator":
        if constants.DOMINATOR_TYPE == "original":
            storage.count_domination = utils.get_dominators(nx_graph_relabeled, constants.STARTING_ACT)
            storage.count_domination = utils.replace_keys(storage.count_domination, mapping_dict)
            utils.calculate_relative_values(storage.count_domination)
            # print(f"Dominators: {storage.count_domination}")
        else:
            raise ValueError("Unknown dominator type")
    elif constants.METHOD == "betweenness":
        for k, v in mapping_dict.items(): # mapping_dict: a dictionary with activity id as keys and activity name as values
            storage.betweenness_centralities[k] = utils.group_betweenness_digraph(nx_graph_relabeled, [v])
        utils.calculate_relative_values(storage.betweenness_centralities)
        # print(storage.betweenness_centralities)

    # -------------------------------------------------------------------------
    sublog_path = constants.DATASET + "_" + "-".join(constants.FILT)
    sample_size = constants.SAMPLE_SIZE

    if constants.APPROXIMATE:
        players = mapping_dict.keys()
    else:
        players = prepare_with_rules(traces_with_timestamps, threshold_range=[(0.85, 0.86)])
        players = [str(x) for x in list(players)]

    print(players)
    g = create_risk_game(players)
    # 0 = full temp deviation, 1 = full betweenness
    # number_range = np.arange(0, 1.05, 0.05)
    # formatted_number_range = [round(num, 2) for num in number_range]
    formatted_number_range = constants.WEIGHTS
    csv_file = open("results/out_with_limited_permutations_syn_data.csv", mode='w', newline='')
    writer = csv.writer(csv_file, delimiter=";")
    writer.writerow(["Weight"] + list(mapping_dict.values()))
    for w in formatted_number_range:
        if w > 0.0:
            eps = constants.BASE_EPS / w
        else:
            eps = constants.BASE_EPS
        # if os.path.isfile(coalition_filepath):
        #     with open(coalition_filepath, "rb") as f:
        #         g.coalitions = pickle.load(f)

        start_time = time.time()
        r1, normalized_r1, avg_samples = calculate_shapley_value_with_limited_permutations(g,variants,  ref_dict=std, nx_graph=nx_graph_relabeled,
                                                                 mapping_dict=mapping_dict, weight=w, e=eps,
                                                                 max_sample_size=constants.SAMPLE_SIZE,
                                                                 approximate=constants.APPROXIMATE)
        overall_time = time.time() - start_time
        print(f"Took {overall_time}s")

        # with open(coalition_filepath, 'wb') as handle:
        #     pickle.dump(g.coalitions, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Sort result
        normalized_r1 = {k: v for k, v in sorted(normalized_r1.items(), key=lambda item: item[1])}
        r1 = {k: v for k, v in sorted(r1.items(), key=lambda item: item[1])}
        for node, shapleys in normalized_r1.items():
            print(
                f"Node '{mapping_dict[str(node)]}' has a normalized impact of {normalized_r1[node]} (absolute: {r1[node]})")

        temp = [w]
        for item in mapping_dict.keys():
            temp.append(round(normalized_r1[item], 4))
        writer.writerow(temp)

        print("Writing colored graph")
        colorize_graph(normalized_r1=normalized_r1, graph=graph, mapping_dict=mapping_dict)

        pydotplus.graphviz.Dot.write(graph,
                                     "results/plots/{0}_{1}_{2}_{3}_{4}_{5}_{6}_graph.png".format(constants.DATASET,
                                                                                                  str(constants.SAMPLE_SIZE),
                                                                                                  str(constants.DEP_THRESH),
                                                                                                  str(w),
                                                                                                  str(avg_samples),
                                                                                                  constants.BASE_EPS,
                                                                                                  "-".join(
                                                                                                      constants.FILT)),
                                     format="png")

        # gather mapping dict with result
        r = Result(r1, normalized_r1, mapping_dict, graph, overall_time)

        save_path = "data/synthetic/{0}_{1}_{2}_{3}_{4}_{5}_{6}.pkl".format(constants.DATASET,
                                                                            str(constants.SAMPLE_SIZE),
                                                                            str(constants.DEP_THRESH),
                                                                            str(w),
                                                                            str(avg_samples),
                                                                            constants.BASE_EPS,
                                                                            "-".join(constants.FILT))

        with open(save_path, 'wb') as handle:
            pickle.dump(r, handle, protocol=pickle.HIGHEST_PROTOCOL)
