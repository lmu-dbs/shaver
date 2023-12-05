from itertools import permutations

import networkx as nx
import pandas as pd

betweeness_normalized = {}  # required for storing group betweenness
degree_normalized = {}  # required for storing group betweenness


def detect_common_antecedents(rules):
    common_antecedents = []
    all_indices = []
    for i, rx in enumerate(rules):
        if i not in all_indices:
            common_pattern = [i]
            for j, ry in enumerate(rules[i + 1:]):
                pos = i + j + 1
                if pos not in all_indices and rx.antecedent == ry.antecedent:
                    common_pattern.append(pos)
                    all_indices.append(pos)
            temp = []
            for x in common_pattern:
                temp.append(rules[x])
            common_antecedents.append(temp)
    return common_antecedents

def convert_log_to_traces_bpic11(log, f):
    # selecting rows based on condition
    result = log[log['case:Diagnosis code'].isin(f)]
    # result = log
    # print(result)

    result = result.groupby('case:concept:name')
    filtered_urgent = []

    pd.set_option('display.expand_frame_repr', False)

    for key in result.groups.keys():
        g = result.get_group(key)
        # Filter for urgent cases
        f = g[g['concept:name'].str.contains("spoed")]
        if not f.empty:  # if f is not empty, at least one event is urgent
            g.sort_values(['time:timestamp'], ascending=True, inplace=True)  # Sort values by 'timestamp'
            g.reset_index(inplace=True, drop=True)
            current_org = g.iloc[0]["org:group"]
            g["time:end_timestamp"] = None
            counter = 0
            for index, row in g.iterrows():
                if current_org != row["org:group"] and counter > 0:
                    g.at[index - counter, "time:end_timestamp"] = g.at[index - 1, "time:timestamp"]
                    current_org = row["org:group"]
                    counter = 0
                elif current_org != row["org:group"] and counter == 0:
                    current_org = row["org:group"]
                    counter = 0
                if index == g.index[-1]:
                    g.at[index - counter, "time:end_timestamp"] = g.at[index, "time:timestamp"]
                counter += 1

            de_dup = g.loc[(g["org:group"].shift() != g["org:group"])]  # Remove consecutive duplicate rows
            de_dup = de_dup.drop('concept:name', axis=1)  # Drop column 'concept:name'
            de_dup = de_dup.rename(columns={'org:group': 'concept:name'})  # Rename column 'org:group' to 'concept:name'
            filtered_urgent.append(de_dup)

    filtered_log = pd.concat(filtered_urgent, axis=0)

    mapping_dict = create_mapping_dict_from_df(filtered_log, 'concept:name')
    inv_map = {v: k for k, v in mapping_dict.items()}

    traces_events = []
    traces_timestamps = []
    traces_end_timestamps = []
    for g in filtered_urgent:
        trace_event = []
        trace_timestamps = []
        trace_end_timestamps = []
        for index, row in g.iterrows():
            act_nr = int(mapping_dict[row['concept:name']])
            ts = row["time:timestamp"]
            ts_end = row["time:end_timestamp"]
            trace_event.append(act_nr)
            trace_timestamps.append(ts)
            trace_end_timestamps.append(ts_end)

        traces_events.append(trace_event)
        traces_timestamps.append(trace_timestamps)
        traces_end_timestamps.append(trace_end_timestamps)

    return filtered_log, traces_events, traces_timestamps, traces_end_timestamps, inv_map


def write_output(title, sequences):
    with open(title, 'w', encoding='UTF8') as f:
        for s in sequences:
            temp = ""
            for el in s:
                temp += str(el) + " -1 "
            temp += "-2"
            if not temp == "-2":
                f.write(temp)
                f.write("\n")


def group_betweenness_digraph(graph, group, normalized=False):
    """
    To calculate the group betweenness as described in "Extending Centrality"
    :param graph: NetworkX Graph, for calculation
    :param group: List of String, list containing the names of the nodes
    :param normalized: boolean indicating if return should be min-max normalized
    :return: betweenness of the group as float
    """
    betweenness = 0.0

    non_group_members = [node for node in graph.nodes if node not in group]
    # non_group_members = [node.obj_dict["attributes"]["label"] for node in graph.nodes if node.obj_dict["attributes"]["label"] not in group]
    for non_group_connection in permutations(non_group_members, 2):
        shortest_paths = 0.0
        shortest_paths_through_group = 0.0
        # print(f"Check between {non_group_connection[0]} and {non_group_connection[1]}")
        try:
            # check all shortest paths
            for current_path in nx.all_shortest_paths(graph, source=non_group_connection[0],
                                                      target=non_group_connection[1]):
                shortest_paths += 1.0
                # Count of paths that pass through the group
                # if all(x in current_path for x in group):
                # Count of paths that intersect the group
                if any(x in current_path for x in group):
                    shortest_paths_through_group += 1.0
            # print(shortest_paths_through_group)
            # print(shortest_paths)
            betweenness += (shortest_paths_through_group / shortest_paths)
        except nx.exception.NetworkXNoPath:
            # print("No path")
            continue

    if normalized:
        betweenness = 1 / ((len(graph.nodes) - len(group)) * (len(graph.nodes) - len(group) - 1))
    return betweenness

def normalize_list(l: list):
    result = []
    min_val = min(l)
    max_val = max(l)
    if min_val == max_val:
        result = [0] * len(l)
    else:
        for el in l:
            result.append((el - min_val) / (max_val - min_val))
    return result


def create_mapping_dict_from_df(df, col: str):
    acts = list(df[col].unique())
    out = dict()
    for i, a in enumerate(acts):
        out[str(a)] = str(i)
    return out
