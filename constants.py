import os
DATASET = "SYNTH"

# FILT = ['M16', '821'] # ovary
# FILT = ['M14', 'M15'] # corpus uteri
# FILT = ['M13', '822'] # cervix uteri
# FILT = ['M13', 'M14', 'M15', 'M16', '821', '822']
FILT = "NA"
DB_PATH = "data/synthetic/synth.pkl"
# DB_PATH = "data/bpic20/InternationalDeclarations..pkl"

COALITION_PATH = DATASET + "_current_coalitions.pkl"
COALITION_FILEPATH = os.path.join("data", "bpic20", COALITION_PATH)

TITLE = "results_" + DATASET
APPROXIMATE = True
VIS_GRANULARITY = 50  # How many shades of red are used in the graph visualization
DEP_THRESH = 0.99  # Dependency threshold for heuristics miner

METHOD = "betweenness"  # betweenness | dominator
DOMINATOR_TYPE = "original"  # immediate | original

STARTING_ACT = "X"  # Only relevant for dominator calculation

BASE_EPS = 0.00001
# WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
WEIGHTS = [0.5]

SAMPLE_SIZE = 10000

