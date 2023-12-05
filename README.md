# ShaVER
This repository contains code which was used to conduct the experiments from the our work 
"ShaVER: Shapley Values to Estimate Multifactorial Risks in Processes"

### Run Experiments
- The `main` in `shaver.py` can be used to run experiments. The settings for a simple example run are already configured.
- Alternatively, parameter values can be set at the beginning
- The constant `APPROXIMATION` is used to decide whether the experiment is conducted with the help of Monte Carlo permutation sampling
or the number of activities is filtered with ER-Miner
- For the latter please ensure that you downloaded [spmf.jar](https://www.philippe-fournier-viger.com/spmf/ERMiner.php) and provided the correct
path in `prepare_players.py`

### Output
- The results are displayed in the console and saved in the corresponding directory in `bpic11` as a pickle file. 
- The resulting colorized plot indicating the risk of certain activities is saved in `results/plots/` 

### Datasets
The [hospital log](https://data.4tu.nl/articles/Real-life_event_logs_-_Hospital_log/12716513) of the BPI Challenge in 2011 by van Dongen was used.
This repository contains preprocessed version of the sub-log containing diagnosis codes M14 and M15. It can be found in `data/bpic11/` as a pickle file.