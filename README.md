# ShaVER
This repository contains code which was used to conduct the experiments from the article 
"ShaVER: Shapley Values to Estimate Multifactorial Risks in Processes"

### Run Experiments
- The `main` in `shaver.py` is used to run experiments. The settings for a simple example run are already configured.
- Alternatively, parameter values can be set in `constants.py`
- The constant `APPROXIMATION` is used to decide whether the experiment is conducted with the help of Monte Carlo permutation sampling
or the number of activities is filtered with ER-Miner
- For the latter please ensure that you downloaded [spmf.jar](https://www.philippe-fournier-viger.com/spmf/ERMiner.php) and provided the correct
path in `prepare_players.py`

### Output
- The results are displayed in the console and saved in the corresponding directory (`bpic11`, `bpic20` or `synthetic`) as a pickle file. 
- The resulting colorized plot indicating the risk of certain activities is saved in `results/plots/`
- The result is saved as a `.csv`-file in the folder `results`

### Datasets
A synthetic log was used which can be found in `data/synthetic`.
Additionally, the [hospital log](https://data.4tu.nl/articles/Real-life_event_logs_-_Hospital_log/12716513) of the BPI Challenge in 2011 by van Dongen was used.
This repository contains preprocessed version of the sub-log containing diagnosis codes M14 and M15. It can be found in `data/bpic11/` as a pickle file.
Furthermore, the PermitLog and the log of InternationalDeclarations was used ([BPI Challenge 2020](https://data.4tu.nl/collections/BPI_Challenge_2020/5065541/1), `data/bpic20`).

### References
1. van Dongen, B. (2011). Real-life event logs - Hospital log (Version 1) [Data set]. Eindhoven University of Technology. https://doi.org/10.4121/UUID:D9769F3D-0AB0-4FB8-803B-0D1120FFCF54
2. van Dongen, Boudewijn (2020): BPI Challenge 2020. Version 1. 4TU.ResearchData. collection. https://doi.org/10.4121/uuid:52fb97d4-4588-43c9-9d04-3604d4613b51
3. Fournier-Viger, P., Lin, C.W., Gomariz, A., Gueniche, T., Soltani, A., Deng, Z., Lam, H. T. (2016). The SPMF Open-Source Data Mining Library Version 2. Proc. 19th European Conference on Principles of Data Mining and Knowledge Discovery (PKDD 2016) Part III, Springer LNCS 9853,  pp. 36-40.
