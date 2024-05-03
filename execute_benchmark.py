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

def full_experiment(use_info, DO, INSPECT, batch_size, hidden_size, output_size, num_epochs, rand_state, results_path, dropout = 0.05, lr = 0.004, nu = 0.1, final_analysis = False):
       
    AM = AlgoModulators(DO, lr = lr, nu = nu)
    dataloader = DO.prep_dataloader(use_info, batch_size)

    model = initialize_model(DO, dataloader, hidden_size, rand_state, dropout = dropout) 

    TVM = TrainValidationManager(use_info, num_epochs, dataloader, batch_size, rand_state, results_path, final_analysis = final_analysis)
    TVM.train_model(DO, AM, model, final_analysis = final_analysis)

    INSPECT.save_train_val_logs(DO, AM, TVM, model, final_analysis = final_analysis)
    
    return DO, TVM, model

def multi_experiments(total_runs, use_info, num_epochs, data_path, inpt_vars, target_vars, miss_vars, hypothesis, partial_perc,
                      INSPECT, batch_size, hidden_size, output_size, results_path, hyperparameters, final_analysis = True):
    
    progress_bar = tqdm(total=total_runs)
    for r_state in range(2000): #
        DO = DataOperator(data_path, inpt_vars, target_vars, miss_vars, hypothesis, partial_perc, r_state, device = "cpu")
        if not DO.lack_partial_coverage:
            full_experiment(use_info, DO, INSPECT, batch_size, hidden_size, output_size, num_epochs, r_state, results_path, hyperparameters["dropout"]["value"], hyperparameters["lr"]["value"], hyperparameters["nu"]["value"], final_analysis)
            progress_bar.update(1)
        if progress_bar.n == total_runs:
            break       
    progress_bar.close()

def benchmark_imputation_sota(data_path, results_path, inpt_vars, target_vars, miss_vars, num_epochs, hypothesis, num_loops, hyperparameters, INSPECT, device = "cpu"):
    
    for imput_method in ["Iterative MICE Imputer", "KNN Imputer", "Miss Forest", "Deep Regressor", "Soft Impute", 
                         "Matrix Factorization", "Hyperimpute"]: #, "Midas"
        counter = 0
        use_info = "use imputation" 
        num_epochs = num_epochs
        for r_state in range(2000):
            DO = DataOperator(data_path, inpt_vars, target_vars, miss_vars, hypothesis, hyperparameters["partial_perc"]["value"], r_state, device = device)
            if not DO.lack_partial_coverage:
                
                #print("###")
                #print(inpt_vars)
                #print("###")
                
                counter += 1
                AM = AlgoModulators(DO, lr = hyperparameters["lr"]["value"])
                IMP = Imputer(DO)
                dataloader = IMP.impute_w_sel(DO, imput_method, hyperparameters["batch_size"]["value"])

                model = initialize_model(DO, dataloader, hyperparameters["hidden_size"]["value"], r_state, hyperparameters["dropout"]["value"] ) 
                TVM = TrainValidationManager(use_info, num_epochs, dataloader, hyperparameters["batch_size"]["value"], r_state, results_path,
                                            imput_method = imput_method, final_analysis = True)
                TVM.train_model(DO, AM, model, final_analysis = True)
                INSPECT.save_train_val_logs(DO, AM, TVM, model, imput_method, final_analysis = True)
            if counter == num_loops:
                break   

def sort_and_retrieve(df, sort_by, column):

    sorted_df = df.sort_values(by=sort_by, ascending=True)
    best_method = sorted_df[column].tolist()[0]

    return best_method            
            
def main():
    
    parser = argparse.ArgumentParser(description="Execute Benchmark for GGH on Input Data")
    
    #parser.add_argument('-data_path', '--data_path', type=str, required=True, help="Path to .csv file with data to use in benchmark.")
    #parser.add_argument('-results_path', '--data_path', type=str, required=True, help="Path to folder where output files and models of the benchmark will be saved.")
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
    

    #Percentage of simulated data with full information
    partial_perc = hyperparameters['partial_perc']['value']
    
    #Model Train Parameters
    hidden_size = hyperparameters['hidden_size']['value']

    batch_size = hyperparameters['batch_size']['value']*len(hypothesis[0])
    output_size = len(target_vars)
    
    #Number of loops to ensure statistical significance
    num_loops = 20

    #Call data, algorithm and model classes
    INSPECT = Inspector(results_path, hidden_size)  
       
    benchmark_imputation_sota(data_path, results_path, inpt_vars, target_vars, miss_vars, int(hyperparameters['epochs']['value']*2.5), hypothesis, num_loops, hyperparameters, INSPECT, device = "cpu")

    multi_experiments(num_loops, "use hypothesis", int(hyperparameters['epochs']['value']), data_path, inpt_vars, target_vars, miss_vars, hypothesis, partial_perc, 
                      INSPECT, batch_size, hidden_size, output_size, results_path, hyperparameters)
    multi_experiments(num_loops, "partial info",   int(hyperparameters['epochs']['value']*2.5), data_path, inpt_vars, target_vars, miss_vars, hypothesis, partial_perc, 
                      INSPECT, batch_size, hidden_size, output_size, results_path, hyperparameters)
    multi_experiments(num_loops, "use known only", int(hyperparameters['epochs']['value']*2.5), data_path, inpt_vars, target_vars, miss_vars, hypothesis, partial_perc,
                      INSPECT, batch_size, hidden_size, output_size, results_path, hyperparameters)
    multi_experiments(num_loops, "full info",      int(hyperparameters['epochs']['value']*2.5), data_path, inpt_vars, target_vars, miss_vars, hypothesis, partial_perc,
                      INSPECT, batch_size, hidden_size, output_size, results_path, hyperparameters)
    
  
    imput_methods_perf = INSPECT.create_comparison_table(final_analysis = True, use_info= "use imputation")
    timestr = time.strftime("%Y%m%d-%H%M")
    imput_methods_perf.to_csv(results_path + f"/use imputation/best_val_performance_{timestr}.csv")

    best_imput = sort_and_retrieve(imput_methods_perf, 'valid_min_errors_mean', 'Method')
    
    results, results_notavg = INSPECT.create_test_comparison_table(data_path, inpt_vars, target_vars, miss_vars, hypothesis, 
                                                                     partial_perc, batch_size, best_imput = best_imput)    
    timestr = time.strftime("%Y%m%d-%H%M")
    results.to_csv(results_path + f"/final_test_performance_{timestr}.csv")
                              
                   
if __name__ == "__main__":
    main()

    
