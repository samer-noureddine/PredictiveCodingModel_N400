# PredictiveCodingModel_N400
Code for the paper:

Nour Eddine, S., Brothers T., Wang L., Spratling, M., Kuperberg G., (under review). A predictive coding model of the N400.
## Abstract
The N400 event-related component has been widely used to investigate the neural mechanisms underlying real-time language comprehension. However, despite decades of research, there is still no unifying theory that can explain both its temporal dynamics and functional properties. In this work, we show that predictive coding – a biologically plausible algorithm for approximating Bayesian inference – offers a promising framework for characterizing the N400. Using an implemented predictive coding computational model, we demonstrate how the N400 can be formalized as the lexico-semantic prediction error produced as the brain infers meaning from linguistic form of incoming words. We show that the magnitude of lexico-semantic prediction error mirrors the functional sensitivity of the N400 to various lexical variables, priming, contextual effects, as well as their higher-order interactions. We further show that the dynamics of the predictive coding algorithm provide a natural explanation for the temporal dynamics of the N400, and a biologically plausible link to neural activity. Together, these findings directly situate the N400 within the broader context of predictive coding research, and suggest that the brain may use the same computational mechanism for inference across linguistic and non-linguistic domains.

## System Requirements

- This code has been tested on Python 3.9.8 in Windows 11, taking roughly 45 minutes to run on a PC with i7-8700 CPU @ 3.20GHz, 3192 Mhz, 6 Cores, 12 Logical Processors, 16 GB RAM. This code is for documentation and reproducibility purposes. A faster version is being developed [here](https://github.com/wmvanvliet/predictive-coding), courtesy of Marijn van Vliet.

### Software Dependencies:
- Python 3.9
- NumPy 1.26.2
- Pandas 1.4.3
- Matplotlib 3.5.2

## Installation Guide
1. **Install Python**: Install Python 3.9 on your system. You can download it from [the official Python website](https://www.python.org/downloads/).
2. **Install Dependencies**: Install the required Python packages (install time is ~5-10 minutes on a normal desktop):
   ```
   pip install -r requirements.txt
   ```

## Reproducing simulations
To reproduce the simulations, navigate to the directory where you have downloaded the files and simply run the following line:
   ```
   python run_simulations.py
   ```
   This will run the simulations and save the results as csv files in `./data/`; and it will reproduce Figures 4, 5, 6, and 8 and save them in `./plots/`. The data and plots have been pre-computed and uploaded to [OSF](https://osf.io/f7upd/?view_only=21226cf9fa9e416e80177242ac17bc72). After generating the simulation csv files, our analyses can be reproduced by running the `./simulation_csv_files/statistical_analysis.R` file in R.
## Expected outputs
If the code runs smoothly, the following plots should be saved in `./plots/`:
![Fig4A](https://github.com/samer-noureddine/PredictiveCodingModel_N400/blob/main/precomputed_plots/Fig4A_ONsize_total_lexsem_PE.png?raw=true)
![Fig4B](https://github.com/samer-noureddine/PredictiveCodingModel_N400/blob/main/precomputed_plots/Fig4B_ONsize_Psd_total_lexsem_PE.png?raw=true)
![Fig4C](https://github.com/samer-noureddine/PredictiveCodingModel_N400/blob/main/precomputed_plots/Fig4C_Frequency_total_lexsem_PE.png?raw=true)
![Fig4D](https://github.com/samer-noureddine/PredictiveCodingModel_N400/blob/main/precomputed_plots/Fig4D_Richness_total_lexsem_PE.png?raw=true)
![Fig5A](https://github.com/samer-noureddine/PredictiveCodingModel_N400/blob/main/precomputed_plots/Fig5A_rep_priming_total_lexsem_err.png?raw=true)
![Fig5B](https://github.com/samer-noureddine/PredictiveCodingModel_N400/blob/main/precomputed_plots/Fig5B_sem_priming_total_lexsem_err.png?raw=true)
![Fig6A](https://github.com/samer-noureddine/PredictiveCodingModel_N400/blob/main/precomputed_plots/Fig6A_cloze_simulations_total_lexsem_err.png?raw=true)
![Fig6B](https://github.com/samer-noureddine/PredictiveCodingModel_N400/blob/main/precomputed_plots/Fig6B_lexical_violation_total_lexsem_err.png?raw=true)
![Fig7A](https://github.com/samer-noureddine/PredictiveCodingModel_N400/blob/main/precomputed_plots/Fig7A_semantic_prediction_overlap_total_lexsem_err.png?raw=true)
![Fig8AB](https://github.com/samer-noureddine/PredictiveCodingModel_N400/blob/main/precomputed_plots/Fig8AB_orthographic_prediction_overlap_total_lexsem_err.png?raw=true)




## Support
For assistance or inquiries, please contact [samer.a.noureddine@gmail.com](mailto:samer.a.noureddine@gmail.com).
