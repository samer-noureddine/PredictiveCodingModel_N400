import os
import pickle
import random
from operator import itemgetter
import numpy as np
import pandas as pd
from scipy import stats

from get_summary import *
from orth_neighborhood_utils import *
from PredictiveCoding_Model import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

random.seed(1)
np.random.seed(1)


def run_simulation(**kwargs):
    # run a simulation and keep only the components needed for plotting and data analysis (to avoid out-of-memory errors)
    full_simulation = Simulation(**kwargs)
    data,fname,sim_input = full_simulation.simulation_data, full_simulation.sim_filename, full_simulation.sim_input
    del full_simulation
    return {"simulation_data" : data,
             "sim_filename" : fname,
             "sim_input": sim_input}


lexicon = Lexicon()
NUM_ITERS = 20
THRESHOLD = 3.0

wordlist = lexicon.words

with open(r'./helper_txt_files/semunrelated_matlab.csv') as csvfile:
    sem_unr = pd.read_csv(csvfile)
    colname = sem_unr.columns[0]
    semunrelated_indices = np.array(sem_unr[colname]) -1

with open(r'./helper_txt_files/unrepeated_matlab.csv') as csvfile:
    unrep = pd.read_csv(csvfile)
    colname = unrep.columns[0]
    unrepeated_indices = np.array(unrep[colname]) -1

with open(r'./helper_txt_files/OrthOverlapStimuli_400stims.csv') as csvfile:
    orth_predoverlap_csv = pd.read_csv(csvfile)
##################### DEFINE STIMULI FOR ALL CONDITIONS ##################### 

standard_stims = wordlist[:512]
np_standard_stims = np.array(standard_stims)

unrepeated_stims = list(np_standard_stims[unrepeated_indices])
# define semantically related pairs
shared_feats =np.dot(lexicon.semfeatmatrix.T, lexicon.semfeatmatrix)
semrelated_indices = np.argwhere(shared_feats == 8)[:,1]
semrelated_stims = list(np_standard_stims[semrelated_indices])
semunrelated_stims = list(np_standard_stims[semunrelated_indices])

##################### LEXICAL EFFECT SIMULATIONS #####################

# Run the simulations for the 512 standard inputs matched on orthographic neighborhood size, frequency and semantic richness
standard_simulation = run_simulation(sim_input = np_standard_stims, clamp_iterations = NUM_ITERS, sim_filename = 'standard_simulation')

# Group simulation data by lexical characteristics for plotting

# high vs low orthographic neighborhood size
standard_words_ONsize =lexicon.ONsize[0,:512]
standard_words_highONsize = standard_words_ONsize > np.median(standard_words_ONsize)
standard_words_lowONsize = np.logical_not(standard_words_highONsize)
# high vs low frequency
standard_words_frequency =lexicon.frequency[0,:512]
standard_words_highfrequency = standard_words_frequency > np.median(standard_words_frequency)
standard_words_lowfrequency = np.logical_not(standard_words_highfrequency)
# high vs low semantic richness
standard_words_highrichness = lexicon.concreteness[:512] == 1
standard_words_lowrichness = np.logical_not(standard_words_highrichness)

num_trials = standard_simulation['simulation_data']['total_lexsem_err'][0].shape[0]
num_iters = standard_simulation['simulation_data']['total_lexsem_err'][0].shape[1]
PRESTIMULUS_WINDOW = 6
final_stim_duration = NUM_ITERS
padding = PRESTIMULUS_WINDOW - 1
full_window = num_iters + padding
PE_vals = np.block([np.ones((num_trials,padding))*0.001, standard_simulation['simulation_data']['total_lexsem_err'][0]])

# high vs low orthographic neighborhood size
data_to_plot_ho = PE_vals[standard_words_highONsize,:].mean(axis = 0)
data_to_plot_lo = PE_vals[standard_words_lowONsize,:].mean(axis = 0)
# high vs low semantic richness
data_to_plot_hc = PE_vals[standard_words_highfrequency,:].mean(axis = 0)
data_to_plot_lc = PE_vals[standard_words_lowfrequency,:].mean(axis = 0)
# high vs low frequency
data_to_plot_hf = PE_vals[standard_words_highrichness,:].mean(axis = 0)
data_to_plot_lf = PE_vals[standard_words_lowrichness,:].mean(axis = 0)


def plot_lexsemPE_by_indices(data_with_high_indices, data_with_low_indices, factor_name):
    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12*1.3, 6*1.3))
    main_ax = fig.add_subplot()
    main_ax.spines.top.set_visible(False)
    main_ax.spines.right.set_visible(False)
    leg_label = 'Lexico-semantic PE'
    main_ax.plot(np.arange(-PRESTIMULUS_WINDOW + 1, final_stim_duration+0.03, 1.0), data_with_high_indices[-full_window:],'k--',label = f'High {factor_name}')
    main_ax.plot(np.arange(-PRESTIMULUS_WINDOW + 1, final_stim_duration+0.03, 1.0), data_with_low_indices[-full_window:],'k',label = f'Low {factor_name}')
    main_ax.set_ylabel(leg_label)
    main_ax.set_xlabel('Iterations')
    main_ax.set_xticks(np.arange(-PRESTIMULUS_WINDOW+1, final_stim_duration+0.03, 5.0))
    main_ax.legend(fontsize = 12)
    main_ax.legend(loc='upper right', frameon=False)
    plt.savefig(f'./plots/{factor_name}_total_lexsem_PE.png')
    plt.savefig(f'./plots/{factor_name}_total_lexsem_PE.svg')

plot_lexsemPE_by_indices(data_to_plot_hc, data_to_plot_lc, 'Richness')
plot_lexsemPE_by_indices(data_to_plot_ho, data_to_plot_lo, 'ONsize')
plot_lexsemPE_by_indices(data_to_plot_hf, data_to_plot_lf, 'Frequency')


# Run the simulations for the 400 "base words" and 400 ONsize-matched pseudowords
word_vs_psd = {"base_word": run_simulation(sim_input = list(orth_predoverlap_csv['BaseWords']), clamp_iterations =NUM_ITERS, sim_filename = 'baseword_firstpresentation'),
               "pseudoword": run_simulation(sim_input = list(orth_predoverlap_csv['PseudowordNeighbors']),clamp_iterations =NUM_ITERS,sim_filename = 'pseudoword_firstpresentation')}

# Group pseudoword simulation data by ONsize for plotting
pseudoword_ONsize = orth_predoverlap_csv['ONsize_PseudowordNeighbors'].to_numpy() 
pseudoword_highON = pseudoword_ONsize > np.median(pseudoword_ONsize)
pseudoword_lowON = np.logical_not(pseudoword_highON)

num_trials = word_vs_psd['pseudoword']['simulation_data']['total_lexsem_err'][0].shape[0]
num_iters = word_vs_psd['pseudoword']['simulation_data']['total_lexsem_err'][0].shape[1]
PRESTIMULUS_WINDOW = 6
final_stim_duration = NUM_ITERS
padding = PRESTIMULUS_WINDOW - 1
full_window = num_iters + padding
PE_vals = np.block([np.ones((num_trials,padding))*0.001, word_vs_psd['pseudoword']['simulation_data']['total_lexsem_err'][0]])

# plot the pseudoword ONsize effect
data_to_plot_psd_ho = PE_vals[pseudoword_highON,:].mean(axis = 0)
data_to_plot_psd_lo = PE_vals[pseudoword_lowON,:].mean(axis = 0)
plot_lexsemPE_by_indices(data_to_plot_psd_ho, data_to_plot_psd_lo, 'ONsize_Psd')


##################### CONTEXTUAL EFFECT SIMULATIONS #####################

# Run the repetition priming simulation
rep_priming = {'unrepeated': run_simulation(sim_input = unrepeated_stims, clamp_iterations =NUM_ITERS, blanks_before_clamp = 2, prevSim = standard_simulation, sim_filename = 'rep_priming_unrepeated'),
               'repeated' : run_simulation(sim_input = standard_stims, clamp_iterations =NUM_ITERS, blanks_before_clamp = 2, prevSim = standard_simulation, sim_filename = 'rep_priming_repeated')}

# Run the semantic priming simulation
sem_priming = {'semunrelated': run_simulation(sim_input = semunrelated_stims, clamp_iterations =NUM_ITERS, blanks_before_clamp = 2, prevSim = standard_simulation, sim_filename = 'sem_priming_semunrelated'),
                'semrelated': run_simulation(sim_input = semrelated_stims, clamp_iterations =NUM_ITERS, blanks_before_clamp = 2, prevSim = standard_simulation, sim_filename = 'sem_priming_semrelated')}

# Run the lexical predictability simulation
cloze_levels = {"low_cloze": 1/lexicon.size, 
                "med_low_cloze": 0.25,
                "med_high_cloze": 0.5,
                "high_cloze": 0.99}

cloze_simulations_preactivate = {}
cloze_simulations_bottomup = {}

for key, val in cloze_levels.items():
    # pre-activate each of the 512 standard inputs from the top down
    cloze_simulations_preactivate.update({key: run_simulation(sim_input = standard_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "top_down", cloze = val, sim_filename = f'cloze_simulations_preact_{key}')})
    # present each of the 512 standard inputs from the bottom up
    cloze_simulations_bottomup.update({key: run_simulation(sim_input = standard_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "bottom_up", prevSim = cloze_simulations_preactivate[key], sim_filename = f'cloze_simulations_bottomup{key}')})

# faster if the data is pre-saved:
# for key, val in cloze_levels.items():
#     # cloze_simulations_preactivate.update({key: run_simulation(sim_input = standard_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "top_down", cloze = val, sim_filename = f'cloze_simulations_preact_{key}')})
#     cloze_simulations_bottomup.update({key: run_simulation(sim_input = standard_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "bottom_up", sim_filename = f'cloze_simulations_bottomup{key}')})

# Run the lexical prediction violation simulation
lexical_violation = {"low_constraint_unexpected": run_simulation(sim_input = unrepeated_stims, clamp_iterations =NUM_ITERS,prevSim = cloze_simulations_preactivate["low_cloze"], sim_filename = 'lexviol_LCunexp'), 
                    "high_constraint_unexpected": run_simulation(sim_input = unrepeated_stims, clamp_iterations =NUM_ITERS,prevSim = cloze_simulations_preactivate["high_cloze"], sim_filename = 'lexviol_HCunexp'),
                    "high_constraint_expected": run_simulation(sim_filename = f'cloze_simulations_bottomuphigh_cloze')
                    }

# Run the anticipatory semantic overlap simulation
semantic_prediction_overlap = {"semunrelated_99cloze": run_simulation(sim_input = semunrelated_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "bottom_up", prevSim = cloze_simulations_preactivate["high_cloze"], sim_filename = 'sempredoverlap_semunrelated_99cloze'),
                                "semrelated_99cloze": run_simulation(sim_input = semrelated_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "bottom_up", prevSim = cloze_simulations_preactivate["high_cloze"], sim_filename = 'sempredoverlap_semrelated_99cloze'),
                                "semunrelated_50cloze": run_simulation(sim_input = semunrelated_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "bottom_up", prevSim = cloze_simulations_preactivate["med_high_cloze"], sim_filename = 'sempredoverlap_semunrelated_50cloze'),
                                "semrelated_50cloze": run_simulation(sim_input = semrelated_stims, clamp_iterations =NUM_ITERS,BU_TD_mode = "bottom_up", prevSim = cloze_simulations_preactivate["med_high_cloze"], sim_filename = 'sempredoverlap_semrelated_50cloze')}
semantic_prediction_overlap.update({"high_constraint_expected": run_simulation(sim_filename = f'cloze_simulations_bottomuphigh_cloze')})

# Pre-activate the set of 400 base words in preparation for the anticipatory orthographic overlap simulation
orthographic_prediction_overlap_preactivation = run_simulation(sim_input = list(orth_predoverlap_csv['BaseWords']), clamp_iterations =NUM_ITERS,BU_TD_mode = "top_down", cloze = 0.99, sim_filename = 'orthpredoverlap_preact')

# Run the anticipatory orthographic overlap simulation for the word neighbors, pseudoword neighbors, word non-neighbors and pseudoword non-neighbors respectively
orth_prediction_overlap_simulation = {orth_predoverlap_csv.keys()[i] : run_simulation(sim_input = list(orth_predoverlap_csv[orth_predoverlap_csv.keys()[i]]),clamp_iterations =NUM_ITERS, prevSim = orthographic_prediction_overlap_preactivation, sim_filename = f'orthpredoverlap_bottomup_{orth_predoverlap_csv.keys()[i]}') for i in [1,2,3,4]}

# # Separate out the word neighbors and word non-neighbors in order to analyze their threshold crossing accuracy
# orth_prediction_overlap_simulation_words = {}
# orth_prediction_overlap_simulation_words['WordNeighbors'] = {}
# orth_prediction_overlap_simulation_words['WordNonNeighbors'] = {}
# orth_prediction_overlap_simulation_words['WordNeighbors'].update(orth_prediction_overlap_simulation['WordNeighbors'])
# orth_prediction_overlap_simulation_words['WordNonNeighbors'].update(orth_prediction_overlap_simulation['WordNonNeighbors'])
# simulation_accuracy_info(orth_prediction_overlap_simulation_words,20)

# # Note that presenting the input for 40 iterations allows 91% accuracy.
#orth_prediction_overlap_simulation_wordinputs_40iters = {orth_predoverlap_csv.keys()[i] : run_simulation(sim_input = list(orth_predoverlap_csv[orth_predoverlap_csv.keys()[i]]),clamp_iterations =40, prevSim = orthographic_prediction_overlap_preactivation, sim_filename = f'orthpredoverlap_bottomup_40_{orth_predoverlap_csv.keys()[i]}') for i in [1,3]}
# simulation_accuracy_info(orth_prediction_overlap_simulation_wordinputs_40iters,40)

##################### PLOT DATA FROM ALL CONTEXTUAL SIMULATIONS #####################

all_simulations = [rep_priming,sem_priming, cloze_simulations_bottomup, lexical_violation, semantic_prediction_overlap,orth_prediction_overlap_simulation]

def simulation_accuracy_info(the_simulation, num_iters_target_presentation):
    for condition in the_simulation.keys():
        # retrieve the activity of the most active lexical state for each trial, ie target input, at all iterations
        most_active_state_activity_per_trial = the_simulation[condition]['simulation_data']['max_lex_state_activation'][0][:,-num_iters_target_presentation:] #(num_trials,num_iterations)
        # indicate whether the threshold was crossed for each trial
        threshold_was_crossed = np.any(most_active_state_activity_per_trial > THRESHOLD,axis = 1)
        # for each trial, find the iteration at which the threshold was crossed
        threshold_crossing_iteration_per_trial = np.argmax(most_active_state_activity_per_trial > THRESHOLD,axis = 1) [threshold_was_crossed]#(threshold_crossing_trials,20)
        # for each trial, find the identity of the lexical state that crossed the threshold
        most_active_state_identity_per_trial = the_simulation[condition]['simulation_data']['max_lex_state_identity'][0][threshold_was_crossed,-num_iters_target_presentation:]#(threshold_crossing_trials,20)
        identity_of_threshold_crosser = most_active_state_identity_per_trial[np.arange(np.sum(threshold_was_crossed)), threshold_crossing_iteration_per_trial] #(threshold_crossing_trials,)
        # retrieve the identity of the "correct" target that should have crossed the threshold
        target_identity = np.array([lexicon.words.index(w) for w in np.array(the_simulation[condition]['sim_input'])[threshold_was_crossed]])
        # check how many of the lexical states that crossed the threshold matched the correct target state
        print(f'The number of threshold crossers in {the_simulation[condition]["sim_filename"]} that matched the identity was {np.sum(target_identity == identity_of_threshold_crosser)} out of {threshold_was_crossed.shape[0]}, {(np.sum(target_identity == identity_of_threshold_crosser)/threshold_was_crossed.shape[0])*100}%')



def get_max_yval(all_simulations):
    max_yval_st = 0
    max_yval_err = 0
    for simulation in all_simulations:
        for sub_sim in simulation.keys():
            data_to_plot_st = np.mean(simulation[sub_sim]['simulation_data']['max_lex_state_activation'][0],axis = 0).T
            data_to_plot_err = np.mean(simulation[sub_sim]['simulation_data']['total_lexsem_err'][0],axis = 0).T
            if max(data_to_plot_st) >= max_yval_st:
                max_yval_st = max(data_to_plot_st)
            if max(data_to_plot_err) >= max_yval_err:
                max_yval_err = max(data_to_plot_err)
    return max_yval_st,max_yval_err

max_yval_state,max_yval_error = get_max_yval(all_simulations)


def total_error_means(filename, simulation, max_yval, ordered_conditions = [],condition_styles = ['r','k','b','g','r:','k:','b:','g:','r--','k--','b--','g--'],labels = [],legend_loc = 'upper right'):

    plt.rcParams.update({'font.size': 22})
    fig = plt.figure(figsize=(12*1.3, 6*1.3))
    main_ax = fig.add_subplot()
    main_ax.spines.top.set_visible(False)
    main_ax.spines.right.set_visible(False)
    leg_label = 'Total Lexico-semantic PE'
    
    if ordered_conditions == []:
        ordered_conditions = list(simulation.keys())
    for i, (sub_sim, style, label) in enumerate(zip(ordered_conditions, condition_styles, labels)):
        num_trials = simulation[sub_sim]['simulation_data']['total_lexsem_err'][0].shape[0]
        num_iters = simulation[sub_sim]['simulation_data']['total_lexsem_err'][0].shape[-1]
        PRESTIMULUS_WINDOW = 21
        individual_stimulus = num_iters == NUM_ITERS +1
        if individual_stimulus:
            final_stim_duration = NUM_ITERS
            padding = PRESTIMULUS_WINDOW - 1
            full_window = num_iters + padding
            err_vals = np.block([np.ones((num_trials,padding))*0.001, simulation[sub_sim]['simulation_data']['total_lexsem_err'][0]])
        else:
            final_stim_duration = num_iters - NUM_ITERS - 1
            final_stim_duration -= np.remainder(final_stim_duration , NUM_ITERS)
            full_window = final_stim_duration + PRESTIMULUS_WINDOW
            err_vals = simulation[sub_sim]['simulation_data']['total_lexsem_err'][0]

        data_to_plot = np.mean(err_vals,axis = 0).T

        main_ax.plot(np.arange(-PRESTIMULUS_WINDOW + 1, final_stim_duration+0.03, 1.0), data_to_plot[-full_window:],style,label = label)
        main_ax.set_ylabel(leg_label)
        main_ax.set_xlabel('Iterations')
        main_ax.set_xticks(np.arange(-PRESTIMULUS_WINDOW+1, final_stim_duration+0.03, 5.0))
        main_ax.set_yticks(np.arange(0, max_yval+ 10, 50))
        main_ax.set_ylim(-25, max_yval+0.3)

    main_ax.legend(labels, fontsize = 12)
    main_ax.legend(loc=legend_loc, frameon=False)
    plt.savefig(f'./plots/{filename}_total_lexsem_err.png')
    plt.savefig(f'./plots/{filename}_total_lexsem_err.svg')


total_error_means('rep_priming', rep_priming, max_yval_error, ordered_conditions = ['unrepeated','repeated'], labels = ['Non-repeated', 'Repeated'], condition_styles= ['k--','k-'])
simulation_accuracy_info(rep_priming,20)
total_error_means('sem_priming', sem_priming, max_yval_error, ordered_conditions = ['semunrelated','semrelated'],  labels = ['Unrelated', 'Related'], condition_styles= ['k--','k-'])
simulation_accuracy_info(sem_priming,20)
total_error_means('cloze_simulations', cloze_simulations_bottomup, max_yval_error, ordered_conditions = ['high_cloze', 'med_high_cloze', 'med_low_cloze', 'low_cloze'], labels = ['99% Cloze', '50% Cloze','25% Cloze', '0.06% Cloze'], condition_styles = ['k-', 'k--', 'k-.', 'k:'])
simulation_accuracy_info(cloze_simulations_bottomup,20)
total_error_means('lexical_violation', lexical_violation, max_yval_error, ordered_conditions = ['high_constraint_expected', 'high_constraint_unexpected', 'low_constraint_unexpected'], labels = ['99% Cloze', '99% Constraint Unexpected','0.06% Constraint Unexpected'], condition_styles = ['k-', 'r:', 'k:'],legend_loc = 'upper left')
simulation_accuracy_info(lexical_violation,20)
total_error_means('semantic_prediction_overlap', semantic_prediction_overlap, max_yval_error, ordered_conditions = ['high_constraint_expected','semrelated_99cloze', 'semunrelated_99cloze'],\
     labels = ['Expected', '99% Constraint Related',  '99% Constraint Unrelated'], condition_styles = ['k-', 'k--', 'k:'],legend_loc = 'upper left')
simulation_accuracy_info(semantic_prediction_overlap,20)

total_error_means('orthographic_prediction_overlap', orth_prediction_overlap_simulation, max_yval_error, ordered_conditions = ['PseudowordNonNeighbors','WordNonNeighbors', 'PseudowordNeighbors', 'WordNeighbors'],\
     labels = ['Unrelated Pseudoword', 'Unrelated Word', 'Related Pseudoword','Related Word'], condition_styles = ['r:', 'k:', 'r--', 'k--'],legend_loc = 'upper left')
orth_prediction_overlap_simulation_justwords = {}
orth_prediction_overlap_simulation_justwords['WordNeighbors']  = orth_prediction_overlap_simulation['WordNeighbors']
orth_prediction_overlap_simulation_justwords['WordNonNeighbors']  = orth_prediction_overlap_simulation['WordNonNeighbors']
simulation_accuracy_info(orth_prediction_overlap_simulation_justwords,20)
##################### CREATE CSV DATA FROM ALL SIMULATIONS #####################

def create_simulation_df(simulation, conditions_dict,simulation_name):
    # given a simulation and a conditions_dict, write a CSV file with the right columns
    sub_simulations = list(simulation.keys()) # list of conditions (e.g., unrepeated, repeated)
    number_of_conditions = len(sub_simulations)
    assert conditions_dict['factors'].shape[0] == number_of_conditions # each condition must have a name
    num_of_trials_condition1 = simulation[sub_simulations[0]]['simulation_data']['total_lexsem_err'].shape[1]
    num_iters_condition1 = simulation[sub_simulations[0]]['simulation_data']['total_lexsem_err'].shape[-1]
    for subsim in sub_simulations[1:]:
        num_of_trials_conditionK = simulation[subsim]['simulation_data']['total_lexsem_err'].shape[1]
        num_iters_conditionK = simulation[subsim]['simulation_data']['total_lexsem_err'].shape[-1]
        assert num_of_trials_conditionK == num_of_trials_condition1 # make sure all conditions have an equal number of trials
        assert num_iters_conditionK == num_iters_condition1 # make sure all conditions have an equal number of iterations per trial
    num_trials_per_condition = num_of_trials_condition1
    num_iters_per_trial = num_iters_condition1
    # get a 10-iteration time_window covering iterations 2 through 11, inclusive.
    # Note that -20 is the first iteration of the final word. -20 + 1 is the 2nd iteration; -20 + 11 is the 12th iteration. 
    time_window = np.arange(num_iters_per_trial)[-NUM_ITERS + 1:-NUM_ITERS + 11] 

    WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold = np.zeros((num_trials_per_condition*number_of_conditions,6))
    # WordInds_LexSemErr_ThresholdCrossingIteration_ONsize_Frequency_Concreteness_ItemCrossedThreshold = np.zeros((num_trials_per_condition*number_of_conditions,7))
    sim_input_all = []
    for condition_number, subsim in enumerate(sub_simulations):
        # retrieve start and end inds in the data matrix
        # confirm that these are the right ones.
        start_ind = num_trials_per_condition*condition_number 
        end_ind = num_trials_per_condition*(condition_number +1)

        sim_input_inds = np.array(get_correct_inds(simulation[subsim]['sim_input'] ,lexicon.words))
        sim_input_all.extend(simulation[subsim]['sim_input'])
        LexSemErr_FullTimeCourse = simulation[subsim]['simulation_data']['total_lexsem_err'][0]
        mean_LexSemErr = np.mean(LexSemErr_FullTimeCourse[:,time_window], axis = 1)

        # create table and add each column one by one
        WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold[start_ind:end_ind,1] = mean_LexSemErr
        boolean_threshold_crossing = simulation[subsim]['simulation_data']['max_lex_state_activation'][0][:,-NUM_ITERS:] > THRESHOLD
        WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold[start_ind:end_ind,5][boolean_threshold_crossing[:,-1]] = 1
        if not np.any(np.isnan(sim_input_inds)):
            WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold[start_ind:end_ind,0] = sim_input_inds
            WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold[start_ind:end_ind,2] = lexicon.ONsize.T[sim_input_inds][:,0]
            WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold[start_ind:end_ind,3] = lexicon.frequency.T[sim_input_inds][:,0]
            conc = np.array([lexicon.concreteness]).T
            conc[conc == 0] = -1 # code nonrich items as -1
            WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold[start_ind:end_ind,4] = conc[sim_input_inds][:,0]
        else:
            WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold[start_ind:end_ind,0] = np.nan
            orth_overlap = np.dot(wordlist_to_orth(simulation[subsim]['sim_input']).T, lexicon.orthmatrix) # find the raw orthographic overlap
            WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold[start_ind:end_ind,2] = np.sum(orth_overlap == 3, axis = 1) # retrieve orthographic neighbors relative to the model's lexicon
            WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold[start_ind:end_ind,3:5] = np.nan
        

    df1 = pd.DataFrame(WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold, columns = 'WordInds_LexSemErr_ONsize_Frequency_Concreteness_ItemCrossedThreshold'.split('_'))
    
    if simulation_name != 'Standard_Simulation':
        cond_names = np.vstack([np.tile(conditions_dict['factors'][i],(num_trials_per_condition,1)) for i in range(number_of_conditions)])
        cond_codes = np.vstack([np.tile(conditions_dict['coding'][i],(num_trials_per_condition,1)) for i in range(number_of_conditions)])
        df2 = pd.DataFrame(cond_names, columns = [i+'_name' for i in conditions_dict['col_names']])
        df3 = pd.DataFrame(cond_codes, columns = [i+'_code' for i in conditions_dict['col_names']])
        final_df = pd.concat([df1,df2,df3], axis=1)
    else:
        final_df = df1
    final_df['Word'] = sim_input_all
    final_df.to_csv(f'./simulation_csv_files/{simulation_name}_N400_{time_window[0]}_to_{time_window[-1]}.csv', index = False)


##### STANDARD SIMULATION #####
# add dummy condition so that it can be passed into `create_simulation_df`
standard_simulation_withdummycondition = {}
standard_simulation_withdummycondition['dummy'] = standard_simulation
std_sim_conditions = {}
std_sim_conditions['col_names'] = np.array(['Dummy'])
std_sim_conditions['factors'] = np.array([['dummy']])
std_sim_conditions['coding'] =  np.array([[0]])
create_simulation_df(standard_simulation_withdummycondition,std_sim_conditions,'Standard_Simulation')


##### LEXICAL STATUS SIMULATION #####

word_vs_psd_conditions = {}
word_vs_psd_conditions['col_names'] = np.array(['IsWord'])
word_vs_psd_conditions['factors'] = np.array([['Word'],['Pseudoword']])
word_vs_psd_conditions['coding'] =  np.array([[1],[-1]])
create_simulation_df(word_vs_psd,word_vs_psd_conditions,'Word_vs_Pseudoword_Simulation')


##### REPETITION PRIMING #####

rep_priming_conditions = {}
rep_priming_conditions['col_names'] = np.array(['Repeated'])
rep_priming_conditions['factors'] = np.array([[i] for i in list(rep_priming.keys())])
rep_priming_conditions['coding'] =  np.array([[-1],[1]])
create_simulation_df(rep_priming,rep_priming_conditions,'RepetitionPriming_Simulation')


##### SEMANTIC PRIMING #####

sem_priming_conditions = {}
sem_priming_conditions['col_names'] = np.array(['SemanticRelatedness'])
sem_priming_conditions['factors'] = np.array([[i] for i in list(sem_priming.keys())])
sem_priming_conditions['coding'] =  np.array([[-1],[1]])
create_simulation_df(sem_priming,sem_priming_conditions,'SemanticPriming_Simulation')


##### LEXICAL PREDICTABILITY #####

cloze_conditions = {}
cloze_conditions['col_names'] = np.array(['Cloze'])
cloze_conditions['factors'] = np.array([[i] for i in list(cloze_simulations_bottomup.keys())])
cloze_conditions['coding'] =  np.array([[1/1579],[0.25],[0.5],[0.99]])
create_simulation_df(cloze_simulations_bottomup,cloze_conditions,'ClozeProbability_Simulation')


##### LEXICAL PREDICTION VIOLATION #####

lexical_violation_conditions = {}
lexical_violation_conditions['col_names'] = np.array(['Constraint', 'IsExpected'])
lexical_violation_conditions['factors'] = np.array([['LowConstraint','Unexpected'],['HighConstraint','Unexpected'], ['HighConstraint','Expected']])
lexical_violation_conditions['coding'] =  np.array([[-1,-1],[1,-1],[1,1]])
create_simulation_df(lexical_violation,lexical_violation_conditions,'LexicalViolation_Simulation')


##### ANTICIPATORY SEMANTIC OVERLAP #####
del semantic_prediction_overlap['high_constraint_expected']

semantic_prediction_overlap_conditions = {}
semantic_prediction_overlap_conditions['col_names'] = np.array(['Relatedness', 'Cloze'])
semantic_prediction_overlap_conditions['factors'] = np.array([['SemUnrelated', 'HighCloze'],['SemRelated', 'HighCloze'], ['SemUnrelated', 'MedCloze'],['SemRelated', 'MedCloze']])
semantic_prediction_overlap_conditions['coding'] =  np.array([[-1,0.99],[1,0.99], [-1,0.5],[1,0.5]])
create_simulation_df(semantic_prediction_overlap,semantic_prediction_overlap_conditions,'SemanticPredictionOverlap_Simulation')


##### ANTICIPATORY ORTHOGRAPHIC OVERLAP #####
orth_prediction_overlap_conditions = {}
orth_prediction_overlap_conditions['col_names'] = np.array(['IsWord', 'IsNeighborofExpected'])
orth_prediction_overlap_conditions['factors'] = np.array([['Word','Neighbor'],['Pseudoword','Neighbor'], ['Word','NonNeighbor'],['Pseudoword','NonNeighbor']])
orth_prediction_overlap_conditions['coding'] =  np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
create_simulation_df(orth_prediction_overlap_simulation,orth_prediction_overlap_conditions,'OrthographicPredictionOverlap_Simulation')
