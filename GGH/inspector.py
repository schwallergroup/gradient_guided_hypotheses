import json
import datetime
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, explained_variance_score, mean_absolute_error, confusion_matrix
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle

from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from .data_ops import DataOperator
from .models import load_model
from .selection_algorithms import normalize

def plot_histogram(df, label_column, histogram_column):
    """
    Plots a histogram using seaborn, color coded based on a label column.
    
    Parameters:
    - df: the DataFrame
    - label_column: the name of the column to use for the labels
    - histogram_column: the name of the column to use for the histogram
    """
    unique_labels = df[label_column].unique()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate the histogram for each unique label
    for label in unique_labels:
        sns.distplot(df[df[label_column] == label][histogram_column], ax=ax, kde=True, label=str(label), bins = 30) #, bins = 30

    # Set the title and labels
    ax.set_title(f'Histogram of {histogram_column} by {label_column}', fontsize=15)
    ax.set_xlabel(histogram_column, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    #plt.xlim([0, 0.005])
    
    # Add a legend
    ax.legend(title=label_column)

    # Display the plot
    plt.show()
    
def plot_confusion_matrix(true_labels, pred_labels):
    # Creating a confusion matrix
    cf_matrix = confusion_matrix(true_labels, pred_labels)

    # Creating a heatmap for the confusion matrix
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

    # Labeling the axes
    plt.xlabel('Predicted Class', fontsize=16)
    plt.ylabel('True Class', fontsize=16)

    # Adding ticks
    plt.xticks([0.5, 1.5], ['Unaltered', 'Noisy'], va='center')
    plt.yticks([0.5, 1.5], ['Unaltered', 'Noisy'], va='center')

    # Title
    plt.title('Confusion Matrix', fontsize=20)

    # Annotations for TP, FN, FP, TN
    labels = ['TP', 'FN', 'FP', 'TN']
    label_positions = [(0, 0), (1, 0), (0, 1), (1, 1)]  # Top-left, Bottom-left, Top-right, Bottom-right

    for label, position in zip(labels, label_positions):
        plt.text(position[0]+0.5, position[1]+0.35, label, ha="center", va="center", 
                 color="black", fontsize=14, weight='bold')

    plt.show()

def visualize_train_val_error(DO, TVM):
    plt.plot(range(len(TVM.train_errors_epoch)), TVM.train_errors_epoch, label='Training Error')
    plt.plot(TVM.val_epochs, TVM.valid_errors_epoch, label='Validation Error')
    plt.legend()
    if TVM.use_info == "full info":
        plt.title(f"Model with {TVM.use_info}")
    else:
        plt.title(f"Model with {TVM.use_info} & {DO.partial_perc} partial full")
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.yscale('log')
    plt.show()

def model_predict(TVM, model, tensors):
    model.eval()
    val_pred = model(tensors)
    val_pred = val_pred.detach().numpy()
    return val_pred
        
class Inspector():
    def __init__(self, save_path, hidden_size):
        self.save_path = save_path
        self.hidden_size = hidden_size


    def save_train_val_logs(self, DO, AM, TVM, model, imput_method = "", final_analysis = False, noise_profile = []):
        
        results_dict = {"number_epochs": TVM.num_epochs, "info_used": TVM.use_info, "perc_full_data": DO.partial_perc, 
                        "n_epochs_no_selection": AM.no_selection_epochs, "Frequency % cutoff": AM.freqperc_cutoff,
                        "hidden_size": self.hidden_size, "random_state": DO.rand_state,
                        "train_errors": TVM.train_errors_epoch, "valid_errors": TVM.valid_errors_epoch, "val_epochs": TVM.val_epochs,
                        "input_data_path": DO.path, "gradwcontext": AM.gradwcontext, "Encompassing Param": AM.nu, "Normalize Grads & Contx": AM.normalize_grads_contx,
                        "cluster_all_classes": AM.cluster_all_classes, "imputation method": imput_method, "model_dropout": model.dropout}
        
        if noise_profile:
            results_dict["DATA_NOISE_PERC"] = noise_profile["DATA_NOISE_PERC"]
            results_dict["NOISE_MINRANGE"] = noise_profile["NOISE_MINRANGE"]
            results_dict["NOISE_MAXRANGE"] = noise_profile["NOISE_MAXRANGE"]
            results_dict["DBSCAN_eps"] = AM.eps_value
            results_dict["DBSCAN_min_samples_ratio"] = AM.min_samples_ratio

        if final_analysis:
            if TVM.use_info != "use imputation":
                final_dir = "/final_analysis"
            else:
                final_dir = f"/final_analysis_{imput_method}"
        else:
            final_dir = ""
        
        if not os.path.exists(self.save_path+f"/{TVM.use_info}"+final_dir):
            os.makedirs(self.save_path+f"/{TVM.use_info}"+final_dir)
        
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d") #-%H-%M-%S-%f
        with open(self.save_path + f"/{TVM.use_info}"+final_dir+f"/ep{TVM.num_epochs}-{time_stamp}_{TVM.rand_state}.json", 'w') as f:
            json.dump(results_dict, f, indent=3)
            
            
    def validation_scatter_r2(self, DO, TVM, model, xmin = -0.05, xmax= 1.05, ymin= -0.05, ymax= 1.05):
    
        val_tensors = self._get_val_tensors(DO, TVM)
        val_pred = model_predict(TVM, model, val_tensors)
        
        #def viz_scatter_r2(val_outcomes, val_pred):
        plt.scatter(DO.df_val[DO.target_vars].values, val_pred)
        plt.xlabel("Correct Outcome")
        plt.ylabel("Predicted Outcome")
        #plt.xlim([xmin,xmax])
        #plt.ylim([ymin,ymax])
        plt.title(f"R2 Score: {round(r2_score(DO.df_val[DO.target_vars].values, val_pred), 3)}")
        plt.show()

    def calculate_val_performance(self, DO, TVM, model):
        val_tensors = self._get_val_tensors(DO, TVM)
        DO.val_predictions = model_predict(TVM, model, val_tensors)
        
        if model.type == "regression":
            r2_score_val = r2_score(DO.df_val[DO.target_vars].values, DO.val_predictions)
            print(f"R2 Score: {round(r2_score_val,3)}")
            return [r2_score_val]  
        else:
            accuracy_score_val = accuracy_score(DO.df_val[DO.target_vars].values, DO.val_predictions.round())
            f1_score_val = f1_score(DO.df_val[DO.target_vars].values, DO.val_predictions.round())
            print(f"Accuracy: {round(accuracy_score_val,3)}")
            print(f"F1_Score: {round(f1_score_val,3)}")
            return [accuracy_score_val, f1_score_val]
    
    def calculate_val_r2score(self, DO, TVM, model, data = "validation"):
        val_tensors = self._get_val_tensors(DO, TVM, data)
        val_pred = model_predict(TVM, model, val_tensors)
        #print(val_pred)
        if data == "validation" or data == "Validation":
            return r2_score(DO.df_val[DO.target_vars].values, val_pred)
        elif data == "test" or data == "Test":
            return r2_score(DO.df_test[DO.target_vars].values, val_pred)
        
    def calculate_val_mse(self, DO, TVM, model, data = "validation"):
        val_tensors = self._get_val_tensors(DO, TVM, data)
        val_pred = model_predict(TVM, model, val_tensors)
        #print(val_pred)
        if data == "validation" or data == "Validation":
            return mean_absolute_error(DO.df_val[DO.target_vars].values, val_pred)
        elif data == "test" or data == "Test":
            return mean_absolute_error(DO.df_test[DO.target_vars].values, val_pred)
        
    def calculate_val_acc(self, DO, TVM, model, data = "validation"):
        val_tensors = self._get_val_tensors(DO, TVM, data)
        val_pred = model_predict(TVM, model, val_tensors)
        if data == "validation" or data == "Validation":
            return accuracy_score(DO.df_val[DO.target_vars].values, val_pred)
        elif data == "test" or data == "Test":
            return accuracy_score(DO.df_test[DO.target_vars].values, val_pred)
    
    def _get_val_tensors(self, DO, TVM, data = "validation"):
        if data == "validation" or data == "Validation":
            if TVM.use_info in ["use hypothesis", "full info", "partial info", "use imputation", "known info noisy simulation", "full info noisy"]:
                return DO.full_val_input_tensor
            elif TVM.use_info in ["use known only"]: 
                return DO.known_val_input_tensor
        elif data == "test" or data == "Test":
            if TVM.use_info in ["use hypothesis", "full info", "partial info", "use imputation", "known info noisy simulation", "full info noisy"]:
                return DO.full_test_input_tensor
            elif TVM.use_info in ["use known only"]: 
                return DO.known_test_input_tensor
        
    
    def _load_saved_results(self, final_analysis=False):
        # Dictionary to store the data
        data_dict = {}

        # List all subdirectories
        subdirectories = [x[0] for x in os.walk(self.save_path)]
        
        for directory in subdirectories:
            
            min_errors = []
            
            if final_analysis and directory.endswith("final_analysis"):
                # Find all json files in the "final_analysis" subdirectory
                json_files = glob.glob(os.path.join(directory, "*.json"))
                if json_files:
                    valid_errors_list = []

                    # Loop through all json files in the "final_analysis" subdirectory
                    for json_file in json_files:
                        with open(json_file) as f:
                            data = json.load(f)
                            val_epochs = data["val_epochs"]
                            valid_errors_list.append(data.get("valid_errors", []))
                            min_errors.append(min(data.get("valid_errors", [])))

                    print(directory)
                    print(len(valid_errors_list))
                    #print(valid_errors_list)

                    transposed_errors = np.transpose(np.array(valid_errors_list))
                    min_valid_errors_mean = []
                    min_valid_errors_std = []
                    for row_idx in range(len(valid_errors_list[0])):
                        col_min = []
                        for col_idx in range(len(valid_errors_list)):
                            col_min.append(np.min(transposed_errors[:row_idx+1, col_idx]))
                        min_valid_errors_mean.append(np.mean(col_min))
                        min_valid_errors_std.append(np.std(col_min))

                    # Average the valid_errors
                    valid_errors_mean = np.mean(np.array(valid_errors_list), axis=0).tolist()
                    valid_errors_std = np.std(np.array(valid_errors_list), axis=0).tolist()
                    valid_errors_20 = np.percentile(np.array(valid_errors_list), 15, axis=0).tolist()
                    valid_errors_25 = np.percentile(np.array(valid_errors_list), 25, axis=0).tolist()
                    valid_errors_30 = np.percentile(np.array(valid_errors_list), 35, axis=0).tolist()
                    valid_errors_45 = np.percentile(np.array(valid_errors_list), 40, axis=0).tolist()
                    valid_errors_50 = np.percentile(np.array(valid_errors_list), 50, axis=0).tolist()
                    valid_errors_55 = np.percentile(np.array(valid_errors_list), 60, axis=0).tolist()
                    valid_errors_70 = np.percentile(np.array(valid_errors_list), 65, axis=0).tolist()
                    valid_errors_75 = np.percentile(np.array(valid_errors_list), 75, axis=0).tolist()
                    valid_errors_80 = np.percentile(np.array(valid_errors_list), 85, axis=0).tolist()

                    # Add to data_dict
                    data_dict[directory.split(os.sep)[-2]] = {"val_epochs": val_epochs,
                                                              "valid_errors_mean": valid_errors_mean, 
                                                              "valid_errors_std": valid_errors_std,
                                                              "valid_min_errors": min_errors,
                                                              "valid_min_errors_mean": np.mean(min_errors),
                                                              "valid_min_errors_std": np.std(min_errors),
                                                              "valid_min_earlystop_errors_mean": min_valid_errors_mean,
                                                              "valid_min_earlystop_errors_std": min_valid_errors_std,
                                                              "valid_errors_20": valid_errors_20,
                                                              "valid_errors_25": valid_errors_25,
                                                              "valid_errors_30": valid_errors_30,
                                                              "valid_errors_45": valid_errors_45,
                                                              "valid_errors_50": valid_errors_50,
                                                              "valid_errors_55": valid_errors_55,
                                                              "valid_errors_70": valid_errors_70,
                                                              "valid_errors_75": valid_errors_75,
                                                              "valid_errors_80": valid_errors_80,
                                                             }
            elif not final_analysis:
                # Find all json files in the subdirectory
                json_files = glob.glob(os.path.join(directory, "*.json"))
                if json_files:
                    # Sort the files by modification time
                    json_files.sort(key=os.path.getmtime)

                    # Open the most recent json file
                    with open(json_files[-1]) as f:
                        # Load the json file and add its content to the dictionary
                        data_dict[directory.split(os.sep)[-1]] = json.load(f)

        return data_dict

    def _load_imputation_results(self, final_analysis):
        # Dictionary to store the data
        data_dict = {}

        if final_analysis:
            # Identify the folders with the naming pattern "final_analysis_method"
            method_folders = glob.glob(os.path.join(self.save_path + "/use imputation", "final_analysis_*"))

            for folder in method_folders:
                method = folder.split("\\")[-1].split("_")[-1]  # Extract method name
                json_files = glob.glob(os.path.join(folder, "*.json"))
                
                min_errors = []
                valid_errors_list = []

                # Loop through all json files in the "final_analysis" subdirectory
                for json_file in json_files:
                    with open(json_file) as f:
                        data = json.load(f)
                        valid_errors = data.get("valid_errors", [])
                        valid_errors_list.extend(valid_errors)
                        val_epochs = data["val_epochs"]
                        min_errors.append(min(valid_errors))

                valid_errors_mean = np.mean(valid_errors_list)
                valid_errors_std = np.std(valid_errors_list)

                data_dict[method] = {
                    "val_epochs": val_epochs,
                    "valid_errors_mean": valid_errors_mean,
                    "valid_errors_std": valid_errors_std,
                    "valid_min_errors": min_errors,
                    "valid_min_errors_mean": np.mean(min_errors),
                    "valid_min_errors_std": np.std(min_errors)
                }
        else:
            json_files = glob.glob(os.path.join(self.save_path+"/use imputation", "*.json"))
            if json_files:
                for json_file in json_files:
                    with open(json_file) as f:
                        data_dict[json_file.split("\\")[-1]] = json.load(f)

        return data_dict    
    

    def create_comparison_table(self, final_analysis = True, **kwargs):
        if "use_info" in kwargs and kwargs["use_info"] == "use imputation":
            all_saved_results = self._load_imputation_results(final_analysis)
        else:
            all_saved_results = self._load_saved_results(final_analysis)
        
        table_data = []

        for key, values in all_saved_results.items():
            row = {
                "Method": key,
                "valid_min_errors_mean": values["valid_min_errors_mean"],
                "valid_min_errors_std": values["valid_min_errors_std"]
            }
            table_data.append(row)

        df = pd.DataFrame(table_data)
        return df

    
    def create_test_comparison_table(self, data_path, inpt_vars, target_vars, miss_vars, hypothesis, partial_perc, batch_size, best_imput = "", use_info = "", noise_profile = {}):
        subdirectories = [x[0] for x in os.walk(self.save_path)]
        all_test_results = {}
        all_test_results_notavg = {}
        for directory in subdirectories:
            r2_scores = []
            mean_squared_errors = []
            mean_absolute_errors = []
            explained_variance_scores = []
            if directory.endswith("final_analysis") or directory.endswith(f"final_analysis_{best_imput}"):
                weights_files = glob.glob(os.path.join(directory, "*.pt"))
                #model_files = glob.glob(os.path.join(directory, "*.pth"))  
                if weights_files:
                    for weights_f in weights_files: #, model_f zip( , model_files)
                        #if weights_f.split(".")[0] != model_f.split(".")[0]:
                        #    raise ValueError("Weights and model don't match.")
                        rand_state = int(weights_f.split("/")[-1].split(".")[0])
                        if use_info not in ["known info noisy simulation", "full info noisy"]:
                            DO = DataOperator(data_path, inpt_vars, target_vars, miss_vars, hypothesis,
                                              partial_perc, rand_state, device = "cpu")
                        else:
                            DO = DataOperator(data_path, inpt_vars, target_vars, miss_vars, hypothesis,
                                              partial_perc, rand_state, device = "cpu", use_case = "noise detection")
                            DO.simulate_noise(noise_profile["DATA_NOISE_PERC"], noise_profile["NOISE_MINRANGE"], noise_profile["NOISE_MAXRANGE"])
                        if ((use_info == "known info noisy simulation" ) or (use_info == "full info noisy")) and (("known info noisy simulation" in weights_f) or ("full info noisy" in weights_f) or ("full info" in weights_f)) \
                            or ((use_info != "known info noisy simulation") and (use_info != "full info noisy")):
                            #print(use_info)
                            #print(weights_f)
                            #print("#2")
                            model = load_model(DO, weights_f, batch_size)  
                            model.eval()
                            if directory.split("/")[-2] in ["full info", "partial info", "use imputation", "use hypothesis", "known info noisy simulation", "full info noisy"]:
                                test_predictions = model(DO.full_test_input_tensor)
                            if directory.split("/")[-2] == "use known only":
                                test_predictions = model(DO.known_test_input_tensor)
                            #print(r2_score(test_predictions.detach().numpy(), DO.df_test[DO.target_vars].values))
                            r2_scores.append(r2_score(DO.df_test[DO.target_vars].values, test_predictions.detach().numpy()))
                            mean_squared_errors.append(mean_squared_error(test_predictions.detach().numpy(), DO.df_test[DO.target_vars].values))
                            mean_absolute_errors.append(mean_absolute_error(test_predictions.detach().numpy(), DO.df_test[DO.target_vars].values))
                            explained_variance_scores.append(explained_variance_score(DO.df_test[DO.target_vars].values, test_predictions.detach().numpy()))

                all_test_results[directory.split("/")[-2]] = {#"avg_cap_r2_score":np.mean([0 if r < 0 else r for r in r2_scores]),
                                                              "avg_r2_score":np.mean(r2_scores),
                                                              "std_r2_score":np.std(r2_scores),
                                                              "avg_mse":np.mean(mean_squared_errors),
                                                              "std_mse":np.std(mean_squared_errors),
                                                              "avg_mae":np.mean(mean_absolute_errors),
                                                              "std_mae":np.std(mean_absolute_errors),
                                                              "explained_variance_scores":np.mean(explained_variance_scores)}
                
                all_test_results_notavg[directory.split("/")[-2]] = {#"avg_cap_r2_score":[0 if r < 0 else r for r in r2_scores],
                                                                      "avg_r2_score":r2_scores,
                                                                      "avg_mse":mean_squared_errors,
                                                                      "avg_mae":mean_absolute_errors,
                                                                      "explained_variance":explained_variance_scores}
        
        df = format2table(all_test_results)
        df_notavg = format2table(all_test_results_notavg)
             
        return df, df_notavg                      
                        
                        
    def display_validation_results(self, final_analysis = False, early_stop = False, **kwargs):
        if "use_info" in kwargs and kwargs["use_info"] == "use imputation":
            all_saved_results = self._load_imputation_results()
        else:
            all_saved_results = self._load_saved_results(final_analysis)
            

        # Initialize variables for y-axis limit
        min_loss = float('inf')
        max_loss = float('-inf')
        
        if final_analysis:
            
            plt.style.use('seaborn-whitegrid')
            plt.figure(figsize=(10, 6))
            colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

            for idx, (key, values) in enumerate(all_saved_results.items()):
                if "valid_errors_mean" in values.keys():
                    min_loss = min(min_loss, np.min([v1-v2 for v1,v2 in zip(values["valid_errors_mean"], values["valid_errors_std"])]))
                    max_loss = max(max_loss, np.max([v1+v2 for v1,v2 in zip(values["valid_errors_mean"], values["valid_errors_std"])]))
                    if "percentile" in kwargs:
                        if kwargs["percentile"] == 25:
                            plt.plot(values["val_epochs"], values["valid_errors_25"], label=f'{key} Val Loss', color=colors[idx], linewidth=2)
                            plt.fill_between(values["val_epochs"], values["valid_errors_20"], values["valid_errors_30"], color=colors[idx], alpha=0.2)
                        elif kwargs["percentile"] == 50:
                            plt.plot(values["val_epochs"], values["valid_errors_50"], label=f'{key} Val Loss', color=colors[idx], linewidth=2)
                            plt.fill_between(values["val_epochs"], values["valid_errors_45"], values["valid_errors_55"], color=colors[idx], alpha=0.2)
                        elif kwargs["percentile"] == 75:
                            plt.plot(values["val_epochs"], values["valid_errors_75"], label=f'{key} Val Loss', color=colors[idx], linewidth=2)
                            plt.fill_between(values["val_epochs"], values["valid_errors_70"], values["valid_errors_80"], color=colors[idx], alpha=0.2)
                   
                    else:
                        if early_stop:
                            plt.plot(values["val_epochs"], values["valid_min_earlystop_errors_mean"], label=f'{key} Val Loss', color=colors[idx], linewidth=2)
                        #plt.plot(values["val_epochs"], values["valid_min_earlystop_errors_mean"], label=f'{key} Val Loss', color=colors[idx], linewidth=2)
                        #plt.fill_between(values["val_epochs"], [i-j**2 for i,j in zip(values["valid_min_earlystop_errors_mean"], values["valid_min_earlystop_errors_std"])], 
                        #                 [i+j**2 for i,j in zip(values["valid_min_earlystop_errors_mean"], values["valid_min_earlystop_errors_std"])]
                        #                 , color=colors[idx], alpha=0.2)
                        else:
                            plt.plot(values["val_epochs"], values["valid_errors_mean"], label=f'{key} Val Loss', color=colors[idx], linewidth=2)
                        #plt.fill_between(values["val_epochs"], [i-j for i,j in zip(values["valid_errors_mean"], values["valid_errors_std"])], 
                        #                 [i+j for i,j in zip(values["valid_errors_mean"], values["valid_errors_std"])], color=colors[idx], alpha=0.2)
                        #plt.fill_between(values["val_epochs"], values["valid_errors_25"], values["valid_errors_75"], color=colors[idx], alpha=0.2)                    

                    
                    
                    
        else:
            for key, values in all_saved_results.items():
                plt.plot(values["val_epochs"], values["valid_errors"], label=f'{key.split(os.sep)[-1]} Val Loss')

 
                                   
        plt.legend(bbox_to_anchor=(1.04, 1), fontsize=10, loc="upper left")
        plt.title(f"Loss Comp. {self.save_path.split('/')[-1]}", fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel("Validation Loss", fontsize=12)
        if ("y_min" in kwargs) and ("y_max" in kwargs):
            plt.ylim([kwargs["y_min"], kwargs["y_max"]])
        else:
            plt.ylim([min_loss - 0.05, max_loss + 0.05])
        plt.grid(True)
        if "save_path" in kwargs:
            plt.savefig(kwargs["save_path"], dpi=300, bbox_inches='tight')
        plt.show()
        
        #return all_saved_results

def format2table(all_test_results):
    table_data = []
    
    # Get the column names from the first key's values dictionary
    first_key = next(iter(all_test_results))
    columns = list(all_test_results[first_key].keys())
    
    for key, values in all_test_results.items():
        row = {"Method": key}
        for column in columns:
            row[column] = values.get(column, None)
        table_data.append(row)

    df = pd.DataFrame(table_data)    
    return df

def corr_vs_inco_sel_freq(DO, kde = False):
    
    sel_hyp_tracker = DO.df_train_hypothesis.final_sel_hyp.values
    hyp_global_true_ids = DO.df_train_hypothesis[DO.df_train_hypothesis["correct_hypothesis"]==True]["global_id"].values
    hyp_global_incorrect_ids = DO.df_train_hypothesis[DO.df_train_hypothesis["correct_hypothesis"]==False]["global_id"].values
    
    correct_sel_freq = [sum(sel_hyp_tracker[global_id]) for global_id in hyp_global_true_ids]
    incorrect_sel_freq = [sum(sel_hyp_tracker[global_id]) for global_id in hyp_global_incorrect_ids]
    
    if kde:
        sns.kdeplot(correct_sel_freq, shade=True, color='green', label="Correct Hypothesis")
        sns.kdeplot(incorrect_sel_freq, shade=True, color='orange', label="Incorrect Hypothesis")
    else:
        plt.hist(correct_sel_freq, density=True, alpha=0.85, bins=20, histtype = "step", color='green', label="Correct Hypothesis")
        plt.hist(incorrect_sel_freq, density=True, alpha=0.85, bins=20, histtype = "step", color='orange', label="Incorrect Hypothesis")        
    plt.xlabel("Selection Frequency")
    plt.ylabel("Hypothesis Probability")
    plt.title(f"Hypothesis Sel. Distributions")
    plt.legend()
    plt.show()

    
def selection_histograms(DO, TVM, num_epochs, rand_state, partial_perc, bins=20, hist_alpha=0.3, kde=True):
    """
    Plots comparative histograms and kernel density estimations for three datasets.
    
    Parameters:
    - data1, data2: lists of floats, the data for the histograms.
    - bins: int or sequence, the number of bins or the bin edges.
    - hist_alpha: float, the alpha transparency for the histograms.
    - kde: bool, if True, plot the kernel density estimation.
    """
    
    sel_hyp_tracker = DO.df_train_hypothesis.final_sel_hyp.values
    hyp_global_true_ids = DO.df_train_hypothesis[DO.df_train_hypothesis["correct_hypothesis"]==True]["global_id"].values
    hyp_global_incorrect_ids = DO.df_train_hypothesis[DO.df_train_hypothesis["correct_hypothesis"]==False]["global_id"].values
    
    correct_sel_freq = [sum(sel_hyp_tracker[global_id]) for global_id in hyp_global_true_ids]
    incorrect_sel_freq = [sum(sel_hyp_tracker[global_id]) for global_id in hyp_global_incorrect_ids]
    
    data1 = correct_sel_freq
    data2 = incorrect_sel_freq
    
    # Define the bins if not provided
    if isinstance(bins, int):
        data_combined = np.hstack((data1, data2))
        bins = np.linspace(0, max(data_combined), bins+1)
    
    # Calculate the histograms without density normalization
    data1_hist, bins = np.histogram(data1, bins=bins, density=False)
    data2_hist, _ = np.histogram(data2, bins=bins, density=False)
    
    # Normalize the counts to percentage
    data1_percent = (data1_hist / sum(data1_hist)) * 100
    data2_percent = (data2_hist / sum(data2_hist)) * 100

    plt.bar(bins[:-1], data1_percent, width=np.diff(bins), align='edge', alpha=hist_alpha, color='green', label="Correct Hypothesis")
    plt.bar(bins[:-1], data2_percent, width=np.diff(bins), align='edge', alpha=hist_alpha, color='red', label="Incorrect Hypothesis")

    # Then draw the rectangles with edges on top
    for left, height in zip(bins[:-1], data1_percent):
        rect = Rectangle((left, 0), np.diff(bins)[0], height, fill=None, alpha=0.7, edgecolor='black', linewidth=1)
        plt.gca().add_patch(rect)
    for left, height in zip(bins[:-1], data2_percent):
        rect = Rectangle((left, 0), np.diff(bins)[0], height, fill=None, alpha=0.7, edgecolor='black', linewidth=1)
        plt.gca().add_patch(rect)
    
    if kde:
        # Calculate KDE on a dense x-axis spanning the range of the bins
        x_d = np.linspace(0, max(bins), 1000) #
        kde1 = gaussian_kde(data1, bw_method=0.15) #
        kde2 = gaussian_kde(data2, bw_method=0.15) #'silverman'
        
        # Compute the KDE values on the dense x-axis
        kde1_vals = kde1(x_d)
        kde2_vals = kde2(x_d)
        
        # Scale the KDE to match the histogram's height scale
        kde1_scale = max(data1_percent) / max(kde1_vals)
        kde2_scale = max(data2_percent) / max(kde2_vals)
        
        plt.plot(x_d, kde1_vals * kde1_scale, color='green', alpha = 0.7)#, linestyle='--')
        plt.plot(x_d, kde2_vals * kde2_scale, color='red', alpha = 0.7)#, linestyle='--')

    # Set the tick direction outwards and add grid lines
    plt.gca().tick_params(direction='out')
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    plt.legend()
    #plt.title("Hypothesis Selelection Distributions", fontdict={'fontname': 'Garamond', 'fontsize': 14})
    plt.xlabel("Selection Frequency", fontdict={'fontname': 'Cambria', 'fontsize': 10})
    plt.ylabel("Hypothesis Probability", fontdict={'fontname': 'Cambria', 'fontsize': 10})
    
    if not os.path.exists(TVM.save_path+"/figures"):
        os.makedirs(TVM.save_path+"/figures")

    plt.savefig(TVM.save_path+f"/figures/Hypothesis_Selection_Distributions_partial{partial_perc}_epochs{num_epochs}_randst{rand_state}.png", dpi=100)
    
    plt.show()
    
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd

def cluster_and_visualize(tensors, hyp_class, true_labels = None, option_3D = False):

    if option_3D:
        tsne = TSNE(n_components=3, random_state=0)
        tensors_tsne = tsne.fit_transform(tensors)
    else:
        # Use t-SNE to reduce the dimensions of the data for visualization
        tsne = TSNE(n_components=2, random_state=0)
        tensors_tsne = tsne.fit_transform(tensors)

    # Plot the tensors and their clusters
    if type(true_labels) != type(None):
        if option_3D:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Scatter3d(x=tensors_tsne[:, 0], y=tensors_tsne[:, 1], z=tensors_tsne[:, 2], mode='markers',
                                               marker=dict( color=true_labels,                # set color to an array/list of desired values
                                                            colorscale='Viridis',   # choose a colorscale
                                                            opacity=0.8
                                                        ))])
            fig.write_html("results/3d_gradients.html")
            fig.show()

        else:
            cdict = {0: 'orange', 1: 'blue', 2: 'red', 3: 'green'}
            
            fig, ax = plt.subplots()
            for g in np.unique(true_labels):
                idx = np.where(true_labels == g)
                ax.scatter(tensors_tsne[idx, 0], tensors_tsne[idx, 1], c = cdict[g], label = g, alpha= 0.75)
            ax.legend()              
            #cdict = {1: 'red', 2: 'blue', 3: 'green'}
            #plt.scatter(tensors_tsne[:, 0], tensors_tsne[:, 1], c=true_labels, alpha= 0.75)   
    else:
        # Perform K-Means clustering on the input tensors
        kmeans = KMeans(n_clusters=3, random_state=0).fit(tensors)
        labels = kmeans.labels_
        plt.scatter(tensors_tsne[:, 0], tensors_tsne[:, 1], c=labels)
    plt.title(f"Scatter of {hyp_class} hypothesis class")
    plt.show()
    
    return tensors_tsne[:, 0], tensors_tsne[:, 1], true_labels

def get_gradients(df, epoch):
    
    return df["gradients"].apply(lambda grads: grads[epoch])

def colored_cluster(DO, TVM):
    
    #do_batch_sel = deepcopy(DO.iloc[:TVM.batch_size])
    data_per_class = {}
    for hyp_class in DO.df_train_hypothesis.hyp_class_id.unique():
        do_hyp_class = deepcopy(DO.df_train_hypothesis[DO.df_train_hypothesis["hyp_class_id"]==hyp_class])
        do_hyp_class["gradients"] = get_gradients(DO.df_train_hypothesis[DO.df_train_hypothesis["hyp_class_id"]==hyp_class], epoch = -1)
        data_per_class[hyp_class] = do_hyp_class
        
        do_hyp_class["gradients"]
    
        #cluster_and_visualize(tensors, hyp_class, true_labels = None, option_3D = False)

def get_label(row):
    if row["partial_full_info"] == 1 and row["correct_hypothesis"] == 1:
        return 3
    elif row["partial_full_info"] == 1 and row["correct_hypothesis"] == 0:
        return 2
    elif row["partial_full_info"] == 0 and row["correct_hypothesis"] == 1:
        return 1
    elif row["partial_full_info"] == 0 and row["correct_hypothesis"] == 0:
        return 0    

def process_groups(dataframe, group_columns, separate_column):
    """
    Process groups in a DataFrame based on the values in specified columns.
    For each group, compute the average array and subtract it from each array in the group.
    Add new columns to the DataFrame with the resulting arrays.

    Parameters:
    - dataframe: pandas DataFrame
    - group_columns: list of column names for grouping
    - separate_column: column name containing arrays to be processed

    Returns:
    - Updated DataFrame with new columns
    """

    # Make sure the specified columns exist in the DataFrame
    assert all(col in dataframe.columns for col in group_columns + [separate_column]), "Column not found in DataFrame"

    # Group by the specified columns
    grouped = dataframe.groupby(group_columns)
    result_df = pd.DataFrame()

    for group_name, group_data in grouped:

        arrays = group_data[separate_column].to_numpy()
        average_array = np.mean(arrays, axis=0)

        subtracted_arrays = [array - average_array for array in arrays]
        
        group_data["grads_avg_rmv"] = subtracted_arrays

        result_df = pd.concat([result_df, group_data], ignore_index=True)

    return result_df

def conv_tensors2arrays(list_tuples_tensors):
    """
    Convert a list of tuples of tensors to a list of tuples of NumPy arrays.

    Parameters:
    - list_of_tuples_of_tensors: List of tuples containing tensors.

    Returns:
    - List of tuples containing NumPy arrays.
    """

    # Convert each tuple of tensors to a tuple of NumPy arrays
    list_tuples_arrays = [tuple(np.array(tensor) for tensor in tensor_tuple) for tensor_tuple in list_tuples_tensors]

    return list_tuples_arrays


def create_heatmap(DO, TVM, x, y, label, hyp_class, rand_state, intensity=1, radius=1, resolution=(1000, 1000), min_x=0, max_x=500, min_y=0, max_y=500, x_extra = [], y_extra = [], color_map='custom', legend_pos = 'lower right'):
    def format_tick(x, pos):
        # Determine the order of magnitude of the current tick
        order_of_magnitude = np.floor(np.log10(x)) if x != 0 else 0
        # Adjust the decimal places depending on the magnitude
        decimal_places = max(0, int(1 - order_of_magnitude))
        # Create the format string and return the formatted tick label
        format_string = '{{:.{}f}}'.format(decimal_places)
        return format_string.format(x)
    
    # Create a blank canvas with given resolution
    heatmap = np.zeros(resolution)
    
    # Normalize and scale x, y coordinates
    x_scaled = np.clip((x - min_x) / (max_x - min_x) * resolution[0], 0, resolution[0] - 1).astype(int)
    y_scaled = np.clip((y - min_y) / (max_y - min_y) * resolution[1], 0, resolution[1] - 1).astype(int)
    
    if x_extra:
        x_scaled_extra = np.clip((x_extra - min_x) / (max_x - min_x) * resolution[0], 0, resolution[0] - 1).astype(int)
        y_scaled_extra = np.clip((y_extra - min_y) / (max_y - min_y) * resolution[1], 0, resolution[1] - 1).astype(int)

    # Populate the heatmap
    for (i, j) in zip(x_scaled, y_scaled):
        heatmap[j][i] += intensity

    # Apply Gaussian filter to smooth the heatmap
    heatmap = gaussian_filter(heatmap, sigma=radius)
    heatmap = np.flipud(heatmap)

    # Create a custom color map from the given palette
    if color_map == 'custom':
        if label == "legacy colors":
            colors = [
                    (0.000, 0.200, 0.600), # Darker blue
                    (0.000, 0.400, 0.800), # Dark blue
                    (0.400, 0.600, 0.800), # Blue
                    (0.600, 0.800, 0.941), # Light blue
                    (0.100, 0.750, 0.550), # Green       
                    (1.000, 1.000, 0.800), # Light yellow
                    (1, 1, 1), # White
            ]
        if label == 0 or label == 2:
            colors = [
                    (0.0, 0.0, 0.0),        # Black
                    (0.5, 0.0, 0.0),        # Dark Red
                    (1.0, 0.0, 0.0),        # Red
                    (1.0, 0.5, 0.0),        # Orange
                    (1.0, 0.65, 0.0),       # Light Orange
                    (1.0, 1.0, 0.0)         # Yellow
                ]      
        if label == 1:
            colors = [
                    (0.0, 0.0, 0.0),        # Black
                    (0.0, 0.5, 0.0),        # Dark Green
                    (0.0, 1.0, 0.0),        # Green
                    (0.5, 1.0, 0.5),        # Light Green
                    (0.75, 1.0, 0.0),       # Yellowish Green
                    (1.0, 1.0, 0.0)         # Yellow
                ]
            colors = [
                    (0.0, 0.0, 0.0),         # Black
                    (0.1, 0.1, 0.5),         # Dark Blue
                    (0.1, 0.2, 0.7),         # Medium Dark Blue
                    (0.2, 0.2, 1.0),         # Blue
                    (0.5, 0.75, 1.0),        # Sky Blue
                    (0.8, 0.9, 1.0)          # Very Light Blue
                ]

        n_bins = [3, 6, 10, 14, 17, 21, 24]  # The bins should match the number of colors
        cmap_name = 'my_list'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=len(n_bins))

    else:
        cm = plt.get_cmap(color_map)
    
    # Define the multiplication factor based on the order of magnitude
    max_value = np.max(heatmap)
    order_of_mag = int(np.floor(np.log10(max_value)))
    scale_factor = 10**order_of_mag

    # Define a custom formatter for the colorbar
    def custom_formatter(x, pos):
        # Multiply the tick value by the multiplication factor and format it
        return f"{x / scale_factor:g}"

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = heatmap / scale_factor
    heatmap_image = ax.imshow(heatmap, cmap=cm)
    
    # Overlay 'X' markers
    if label == 0 or label == 2:
        plt.scatter(x_scaled, resolution[1] - y_scaled - 1, color="white", marker="x", s=50, linewidths=0.5, label="∇f+ Incorrect Hypothesis") #'red'
    elif label == 1:
        plt.scatter(x_scaled, resolution[1] - y_scaled - 1, color="white", marker="o", s=70, linewidths=0.5, label="∇f+ Correct Hypothesis") #'green'
    elif label == 3:
        plt.scatter(x_scaled, resolution[1] - y_scaled - 1, color="white", marker='x', s=80, linewidths=2, label="∇f+ Complete Data")  #"black"    #(0.1, 0.5, 0.5)
    if x_extra:
        plt.scatter(x_scaled_extra, resolution[1] - y_scaled_extra- 1, color="white", marker='*', s=180, linewidths=1, label="∇f+ Complete Data") #(0.71,0.56,0.82)

    x_ticks = np.linspace(0, resolution[0] - 1, 5)  
    y_ticks = np.linspace(0, resolution[1] - 1, 5)
    x_tick_labels = np.linspace(0, 100, 5)
    y_tick_labels = np.linspace(0, 100, 5)
    plt.xticks(x_ticks, x_tick_labels, fontsize = 12)
    plt.yticks(y_ticks, y_tick_labels, fontsize = 12) 
    
    num_colors = len(colors)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(heatmap_image, cax=cax)

    # Set the colorbar label
    cbar.set_label('Gradient Density', rotation=270, labelpad=20, fontsize="18")

    # Apply the custom formatter to the color bar
    cbar.set_ticks([np.round(v, 2) for v in np.linspace(heatmap.min(), heatmap.max(), num_colors + 2)]) #/(10**order_of_mag)
    cbar.update_ticks()
    cbar.ax.text(1.30, 1.005, f'$\\times10^{{{int(order_of_mag)}}}$', transform=cbar.ax.transAxes, ha='right', va='bottom')

    # Set ticks for the heatmap
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xlabel('t-SNE 1', fontsize=14)
    ax.set_ylabel('t-SNE 2', fontsize=14)

    if DO.use_case != "noise detection":
        if label == 0 or label == 2:
            legend_elements = [Line2D([0], [0], marker='x', color='k', label="Incorrect Hypothesis ∇f+", markerfacecolor='black', markersize=8, linestyle='None')]
        elif label == 1:
            legend_elements = [Line2D([0], [0], marker='o', color='k', label="Correct Hypothesis ∇f+", markerfacecolor='black', markersize=8, linestyle='None')]        
        if x_extra:
            legend_elements.append(Line2D([0], [0], marker='*', color='k', label="Ground Truth ∇f+", markerfacecolor='black', markersize=14, linestyle='None'))
    else:
        if label == 0:
            legend_elements = [Line2D([0], [0], marker='x', color='k', label="Noisy Data ∇f+", markerfacecolor='black', markersize=8, linestyle='None')]
        elif label == 1:
            legend_elements = [Line2D([0], [0], marker='o', color='k', label="Unaltered Data ∇f+", markerfacecolor='black', markersize=8, linestyle='None')] 

        
    ax.legend(handles=legend_elements, title="Markers", loc=legend_pos, frameon=True, facecolor='w', labelcolor='k', fontsize = 14)
    
    
    if not os.path.exists(TVM.save_path+"/figures"):
        os.makedirs(TVM.save_path+"/figures")
    if DO.use_case != "noise detection":
        if label == 0 or label == 2:
            plt.savefig(TVM.save_path+f"/figures/context_grads_heatmap_incorrect_partial{DO.partial_perc}_hyp_{hyp_class}_randst_{rand_state}.png", dpi=100)
        if label == 1:
            plt.savefig(TVM.save_path+f"/figures/context_grads_heatmap_correct_partial{DO.partial_perc}_hyp_{hyp_class}_randst_{rand_state}.png", dpi=100)
    else:
        if label == 0:
            plt.savefig(TVM.save_path+f"/figures/context_grads_heatmap_noisy_data_randst_{rand_state}_numepochs{TVM.num_epochs}.png", dpi=100)
        if label == 1:
            plt.savefig(TVM.save_path+f"/figures/context_grads_heatmap_unaltered_data_randst_randst_{rand_state}_numepochs{TVM.num_epochs}.png", dpi=100)
        
    plt.show()

    plt.close()
    
    return heatmap, cm

def get_gradarrays_n_labels(DO, hyp_class, layer = -2, remov_avg = True, include_context = True, normalize_grads_context = False, loss_in_context = False, only_loss_context = False, num_batches = 2, epoch = -1, use_case = "hypothesis"):
    
    if use_case == "noise_detection":
        remov_avg = False
        #print("In noise detection there are no groups of Hypothesis Class")
        df_train_last_batch = deepcopy(DO.df_train_noisy.iloc[-DO.batch_size*num_batches:])
        df_train_last_batch = df_train_last_batch[DO.inpt_vars+[DO.target_vars[0]+"_noisy"]+["noise_added", "loss", "gradients"]]       
        df_train_last_batch["gradients"] = get_gradients(DO.df_train_noisy, epoch = epoch)
        
        df_train_last_batch["label"] = df_train_last_batch["noise_added"].values
        
        if remov_avg:
            flattened_arrays = [array.flatten() for array in df_train_last_batch["grads_avg_rmv"].values]
        else:
            flattened_arrays = [array.flatten() for array in df_train_last_batch["gradients"].values]
        combined_array = np.vstack(flattened_arrays)

        inpt_contxt = df_train_last_batch[DO.inpt_vars+[DO.target_vars[0]+"_noisy"]].values
        #using same variable name as the group of hypothesis class to make the code more concise 
        do_hyp_class = df_train_last_batch
    else:
        #Get gradients and datapoints from the last batch 
        df_train_hyp_last_batch = deepcopy(DO.df_train_hypothesis.iloc[-DO.batch_size*num_batches:])
        df_train_hyp_last_batch = df_train_hyp_last_batch[DO.inpt_vars+[miss_var + "_hypothesis" for miss_var in DO.miss_vars]
                                                    +DO.target_vars+["hyp_class_id","partial_full_info","correct_hypothesis", "loss", "gradients"]]
        df_train_hyp_last_batch["gradients"] = get_gradients(DO.df_train_hypothesis, epoch = epoch)

        #Get gradients and datapoints from partial
        do_hyp = deepcopy(DO.df_train_hypothesis)
        do_hyp_part = do_hyp[(do_hyp["partial_full_info"] == 1) & (do_hyp["correct_hypothesis"] == 1)][DO.inpt_vars+[miss_var + "_hypothesis" for miss_var in DO.miss_vars]
                                                                                                       +DO.target_vars+["loss","hyp_class_id","partial_full_info","correct_hypothesis"]]
        do_hyp_part["gradients"] = [grads[layer].reshape(grads[layer].shape[1]) for grads in conv_tensors2arrays(DO.latest_partial_grads)] 

        do_hyp = deepcopy(DO.df_train_hypothesis)
        do_hyp_inc_part = do_hyp[(do_hyp["partial_full_info"] == 1) & (do_hyp["correct_hypothesis"] == 0)][DO.inpt_vars+[miss_var + "_hypothesis" for miss_var in DO.miss_vars]
                                                                                                           +DO.target_vars+["loss","hyp_class_id","partial_full_info","correct_hypothesis"]]
        do_hyp_inc_part["gradients"] = [grads[layer].reshape(grads[layer].shape[1]) for grads in conv_tensors2arrays(DO.latest_inc_partial_grads)] 

        #Combine for visuazation
        df_train_hyp_last_batch_allpart = pd.concat([df_train_hyp_last_batch, do_hyp_part, do_hyp_inc_part]) #df_train_hyp_last600
        do_hyp_class = df_train_hyp_last_batch_allpart[df_train_hyp_last_batch_allpart["hyp_class_id"]==hyp_class]
        do_hyp_class["label"] = do_hyp_class.apply(lambda row: get_label(row), axis = 1)

        do_hyp_class = process_groups(do_hyp_class, DO.inpt_vars+DO.target_vars, "gradients")

        if remov_avg:
            flattened_arrays = [array.flatten() for array in do_hyp_class["grads_avg_rmv"].values] #grads_avg_rmv
        else:
            flattened_arrays = [array.flatten() for array in do_hyp_class["gradients"].values]
        combined_array = np.vstack(flattened_arrays)

        inpt_contxt = do_hyp_class[DO.inpt_vars+[miss_var + "_hypothesis" for miss_var in DO.miss_vars]+DO.target_vars].values
    
    array_grads_context = np.hstack([combined_array, inpt_contxt])
    
    if loss_in_context:
        loss_vert_array = extract_elements(do_hyp_class, 'loss', epoch = epoch)
        if not only_loss_context:
            array_grads_context = np.hstack([array_grads_context, loss_vert_array])
        else:
            array_grads_context = np.hstack([combined_array, loss_vert_array])
        
    
    if normalize_grads_context:
        array_grads_context = normalize(array_grads_context)
        combined_array = normalize(combined_array)
    
    if include_context:
        return array_grads_context, do_hyp_class
    else:
        return combined_array, do_hyp_class

def prep_tsne_min_max_ranges(tsne1, tsne2):
    if min(tsne1) < 0:
        overall_xmin = min(tsne1)*1.15
    else:
        overall_xmin = min(tsne1)/0.85
    if max(tsne1) > 0:
        overall_xmax = max(tsne1)*1.15
    else:
        overall_xmax = max(tsne1)/0.85
    if min(tsne2) < 0:
        overall_ymin = min(tsne2)*1.15
    else:
        overall_ymin = min(tsne2)/0.85
    if max(tsne2) > 0:
        overall_ymax = max(tsne2)*1.15
    else:
        overall_ymax = max(tsne2)/0.85    
    
    return overall_xmin, overall_xmax, overall_ymin, overall_ymax

def sep_grads_by_labels(TVM, tsne1, tsen2, true_labels, use_case = "hypothesis"):
    inc_grads_t1 = []
    inc_grads_t2 = []
    corr_grads_t1 = []
    corr_grads_t2 = []
    corr_part_t1 = []
    corr_part_t2 = []
    
    if use_case == "hypothesis":

        for t1, t2, t_l in zip(tsne1, tsen2, true_labels):
            if t_l == 0 or t_l == 2:
                inc_grads_t1.append(t1)
                inc_grads_t2.append(t2)
            elif t_l == 1:
                corr_grads_t1.append(t1)
                corr_grads_t2.append(t2)
            elif t_l == 3:
                corr_part_t1.append(t1)
                corr_part_t2.append(t2) 

        return inc_grads_t1, inc_grads_t2, corr_grads_t1, corr_grads_t2, corr_part_t1, corr_part_t2
    
    elif use_case == "noise_detection":
        noise_grads_t1 = []
        noise_grads_t2 = []
        for t1, t2, t_l in zip(tsne1, tsen2, true_labels):
            if t_l == 1:
                noise_grads_t1.append(t1)
                noise_grads_t2.append(t2)
            elif t_l == 0:
                corr_grads_t1.append(t1)
                corr_grads_t2.append(t2)
        return noise_grads_t1, noise_grads_t2, corr_grads_t1, corr_grads_t2

def corr_vs_inco_sel_freq(DO, kde = False):
    
    sel_hyp_tracker = DO.df_train_hypothesis.final_sel_hyp.values
    hyp_global_true_ids = DO.df_train_hypothesis[DO.df_train_hypothesis["correct_hypothesis"]==True]["global_id"].values
    hyp_global_incorrect_ids = DO.df_train_hypothesis[DO.df_train_hypothesis["correct_hypothesis"]==False]["global_id"].values
    
    correct_sel_freq = [sum(sel_hyp_tracker[global_id]) for global_id in hyp_global_true_ids]
    incorrect_sel_freq = [sum(sel_hyp_tracker[global_id]) for global_id in hyp_global_incorrect_ids]
    
    if kde:
        sns.kdeplot(correct_sel_freq, shade=True, color='green', label="Correct Hypothesis")
        sns.kdeplot(incorrect_sel_freq, shade=True, color='orange', label="Incorrect Hypothesis")
    else:
        plt.hist(correct_sel_freq, density=True, alpha=0.85, bins=20, histtype = "step", color='green', label="Correct Hypothesis")
        plt.hist(incorrect_sel_freq, density=True, alpha=0.85, bins=20, histtype = "step", color='orange', label="Incorrect Hypothesis")        
    plt.xlabel("Selection Frequency")
    plt.ylabel("Hypothesis Probability")
    plt.title(f"Hypothesis Sel. Distributions")
    plt.legend()
    plt.show()
    
    return correct_sel_freq, incorrect_sel_freq

def extract_elements(df, column_name, epoch = -1):
    # This list will hold the last elements from each row
    elements = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the last element of the list in the specified column
        element = row[column_name][epoch]
        elements.append(element)

    # Convert the list into a NumPy array
    vertical_array = np.array(elements).reshape(-1, 1)

    return vertical_array