from spmf import Spmf

import utils


def prepare_with_rules(traces_with_timestamps, threshold_range):
    traces_events = []

    for t in traces_with_timestamps:
        traces_events.append([x[0] for x in t])
    utils.write_output("output_sequences.txt", traces_events)
    for support, confidence in threshold_range:
        spmf = Spmf('ERMiner',
                    input_filename="output_sequences.txt",
                    output_filename="output_rules.txt",
                    spmf_bin_location_dir="C:/path/to/spmf/",  # insert absolute path to spmf.jar here
                    arguments=[support, confidence])
        spmf.run()
        rules_df = spmf.to_pandas_dataframe()
        patterns = rules_df["pattern"].tolist()
        print(patterns)
        ants = []
        cons = []
        itemsets = []
        for pattern in patterns:
            pattern = pattern.split(" ==> ")
            a = [int(x) for x in pattern[0].split(",")]
            c = [int(x) for x in pattern[1].split(",")]
            i = a + c
            ants.append(a)
            cons.append(c)
            itemsets.append(i)
        rules_df["ant"] = ants
        rules_df["cons"] = cons
        rules_df["itemset"] = itemsets
        rules_df["sup_ant"] = None
        rules_df["sup_cons"] = None
        ant_cons = [(x.split(" ==> ")[0].split(","), x.split(" ==> ")[1].split(",")) for x in patterns]
        ant_cons = [item for sublist in ant_cons for item in sublist]
        sup_dict = dict()
        for l in ant_cons:
            l = [int(x) for x in l]
            # print(f"Getting support for {l}")
            if str(l) not in sup_dict.keys():
                sup_dict[str(l)] = 0
                for trace in traces_events:
                    # print(f"Cecking {trace}")
                    if set(l) <= set(trace):
                        sup_dict[str(l)] += 1
        print(sup_dict)
        for index, row in rules_df.iterrows():
            rules_df.at[index, "sup_ant"] = sup_dict[str(row["ant"])]
            rules_df.at[index, "sup_cons"] = sup_dict[str(row["cons"])]

        rules_df["lift"] = (rules_df.sup/len(traces_events)) / ((rules_df.sup_ant/len(traces_events)) * (rules_df.sup_cons/len(traces_events)))
        print(rules_df[["pattern", "lift"]])
        # filter independent rules
        # rules_df.drop(rules_df[rules_df.lift == 1.0].index, inplace=True)
        rules_df.reset_index(drop=False, inplace=True)
        print(rules_df[["pattern", "itemset", "lift"]])

        all_players = set([x for sublist in rules_df.itemset.tolist() for x in sublist])
        print(all_players)
        return all_players


