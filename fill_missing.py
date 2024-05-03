import argparse
import json

import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

import sys,os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'GGH')
from GGH.data_ops import DataOperator
from GGH.selection_algorithms import AlgoModulators
from GGH.models import initialize_model, Autoencoder
from GGH.train_val_loop import TrainValidationManager
from GGH.inspector import Inspector, plot_histogram, visualize_train_val_error, selection_histograms, create_heatmap, cluster_and_visualize, \
                            get_gradarrays_n_labels, prep_tsne_min_max_ranges, sep_grads_by_labels
from GGH.imputation_methods import Imputer

def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_selection_ratio(final_sel_hyp):
    return sum(final_sel_hyp)/len(final_sel_hyp)
    
def main():
    
    parser = argparse.ArgumentParser(description="Execute Benchmark for GGH on Input Data")
    parser.add_argument('-data_config_path', '--data_cfg_path', type=str, required=False, default="input_config.json", help="Path to JSON file with name of columns for each role.")
    parser.add_argument('-hyperparameters_path', '--hyp_cfg_path', type=str, required=False, default="hyperparameters.json", help="Path to JSON file with all hyperparameters.")

    # Parse the command line arguments
    args = parser.parse_args()
    data_config = load_config(args.data_cfg_path)
    hyperparameters = load_config(args.hyp_cfg_path)
    
    #User requiered data and parameters
    data_path    = data_config['data_path']['value']
    results_path = data_config['results_path']['value']
    
    inpt_vars    = data_config['inpt_vars']['value']
    target_vars  = data_config['target_vars']['value']
    miss_vars    = data_config['miss_vars']['value']
    hypothesis   = data_config['hypothesis']['value']
    
    #Model Train Parameters
    hidden_size = hyperparameters['hidden_size']['value']

    rand_state = 42
    
    batch_size = hyperparameters['batch_size']['value']*len(hypothesis[0])
    output_size = len(target_vars)
    
    DO = DataOperator(data_path, inpt_vars, target_vars, miss_vars, hypothesis, rand_state = rand_state, device = "cpu", use_mode = "fill_missing") #"fill_missing" "validation"
    
    AM = AlgoModulators(DO, lr = hyperparameters["lr"]["value"], nu = hyperparameters["nu"]["value"])
    dataloader = DO.prep_dataloader("use hypothesis", batch_size)

    model = initialize_model(DO, dataloader, hidden_size, rand_state, dropout = hyperparameters["dropout"]["value"]) 

    TVM = TrainValidationManager("use hypothesis", int(hyperparameters['epochs']['value']), dataloader, batch_size, rand_state, results_path, final_analysis = True)
    TVM.train_model(DO, AM, model, final_analysis = True)
    
    missing_hypothesis_ratio = DO.df_train_hypothesis[DO.inpt_vars+DO.miss_vars+DO.target_vars+[name+"_hypothesis" for name in DO.miss_vars]+["global_id", "sel_hyp_tracker", "final_sel_hyp", "loss", "gradients"]]

    missing_hypothesis_ratio["selection_ratio"] = missing_hypothesis_ratio["final_sel_hyp"].apply(get_selection_ratio)
    
    timestr = time.strftime("%Y%m%d-%H%M")
    missing_hypothesis_ratio.to_csv(results_path + f"/selection_ratio_{timestr}.csv")
    
    
                   
if __name__ == "__main__":
    main()