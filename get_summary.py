import numpy as np
from orth_neighborhood_utils import *


def get_correct_inds(stimset, model_lexicon):
    # given a list of strings, return their indices in the model's lexicon (if they exist; otherwise return nan).
    lexicon_ind_list = []
    for item in stimset:
        if item not in model_lexicon:
            lexicon_ind_list.append(np.nan)
        else:
            # grab everything about this item's orthography, lexical and semantic indices
            lexicon_ind_list.append(model_lexicon.index(item))
    return np.array(lexicon_ind_list)

def get_avg_sem(model):
    avg_semfeatmatrix = model.lexicon.semfeatmatrix / np.sum(model.lexicon.semfeatmatrix, axis =0)
    final_mat = np.dot(model.statespace['sem'][0].T, avg_semfeatmatrix).T
    return final_mat


def get_summary(model):
    '''
    Takes in a model at a given iteration and records summary values from the model (e.g. total lexico-semantic PE) at that iteration
    ''' 

    # The first four dimensions of model.statespace[level]: 
    kinds = ['state', 'reconstruction', 'preactivation', 'prediction_error'] 
    
    correct_lexicon_inds = get_correct_inds(model.sim_input, model.lexicon.words)

    summary_dict = {}
    summary_dict['lex_accuracy'] = np.argmax(model.statespace['lex'][0], axis = 0) == correct_lexicon_inds
    summary_dict['sem_accuracy'] = np.argmax(get_avg_sem(model), axis = 0) == correct_lexicon_inds
    
    target_representation = {'orth': model.lexicon.orthmatrix.T,
                                'lex': model.lexicon.lexicalmatrix.T,
                                'sem': model.lexicon.semfeatmatrix.T}
    
    # INDICATE WHAT INFORMATION TO EXTRACT HERE
    summary_dict['max_lex_state_activation'] =  np.max(model.statespace['lex'][0], axis = 0)
    summary_dict['total_lexsem_err'] = np.sum(model.statespace['lex'][3], axis = 0) + np.sum(model.statespace['sem'][3], axis = 0)
    summary_dict['max_lex_state_identity'] =  np.argmax(model.statespace['lex'][0], axis = 0)

    return summary_dict