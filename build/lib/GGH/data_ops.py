from typing import List
import pandas as pd
import numpy as np
import random
import os
from copy import deepcopy

import torch
from torch.utils.data import TensorDataset, DataLoader



class DataOperator():
    def __init__(self, path, inpt_vars, target_vars, miss_vars, hypothesis, partial_perc = 0.2, 
                 rand_state = 42, hyp_class_thresh = [], device = "cpu", use_mode = "validation",
                 process_data = True, verbose = False, pre_defined_train = False, use_case = "hypothesis"):  
              
        self.path = path
        self.inpt_vars = deepcopy(inpt_vars)
        self.miss_vars = deepcopy(miss_vars)
        self.hypothesis = deepcopy(hypothesis)
        self.target_vars = target_vars
        
        self.device = device
        self.rand_state = rand_state
        self.process_data = process_data
        self.verbose = verbose
        
        self._load_prep_df()
        self.hyp_class_thresh = self._set_hyp_class_threshold(deepcopy(hyp_class_thresh))
        self.num_miss_vars = len(self.miss_vars)
        self.use_mode = use_mode
        self.use_case = use_case
        random.seed(self.rand_state)
        
        if use_mode == "validation":
            if type(pre_defined_train) == pd.DataFrame:
                self.df_train = pre_defined_train
            else:
                self.df_train = self.df.iloc[:int(len(self.df)*0.72)]
            self.df_val = self.df.iloc[int(len(self.df)*0.72):int(len(self.df)*0.88)]
            self.df_test = self.df.iloc[int(len(self.df)*0.88):]
        elif use_mode == "fill_missing":
            #self.df_train = self.df.iloc[:]
            #these are placeholders in the case of fill_missing, in that scenario no val/test is expected.
            #self.df_val = deepcopy(self.df)
            self.df_train = self.df.iloc[:int(len(self.df)*0.99)]
            self.df_train.reset_index(inplace=True, drop=True)
            #these are placeholders in the case of fill_missing, in that scenario no val/test is expected.
            self.df_val = self.df.iloc[:int(len(self.df)*0.01)]
            self.df_test = self.df.iloc[:int(len(self.df)*0.01)]

        self.known_input = self.df_train[self.inpt_vars].values
        self.full_input = self.df_train[self.inpt_vars+self.miss_vars].values
        
        if use_mode == "validation":
            self.partial_perc = deepcopy(partial_perc)
            self.partial_rows_id = np.sort(random.sample(range(0, self.full_input.shape[0]-1),
                                                         int(self.full_input.shape[0]*self.partial_perc)))
            print(type(self.partial_rows_id))
            print(self.partial_rows_id)
            
            self.partial_input = self.full_input[self.partial_rows_id, :]
            
        elif use_mode == "fill_missing":
            self.partial_input = deepcopy(self.df_train[self.inpt_vars+self.miss_vars].dropna(subset=self.miss_vars))
            self.partial_perc = len(self.partial_input)/len(self.df)
            self.partial_rows_id = np.asarray(self.partial_input.index.tolist())
            self.partial_input = self.partial_input.values
            
            #print("###########-------###############")
            #print(type(self.partial_rows_id))
            #print(self.partial_rows_id)
            
            #self.partial_perc = deepcopy(partial_perc)
            #self.partial_rows_id = np.sort(random.sample(range(0, self.full_input.shape[0]-1),
            #                                             int(self.full_input.shape[0]*self.partial_perc)))
            #self.partial_input = self.full_input[self.partial_rows_id, :]
        
        self._create_hypothesis_data()
        self.known_val_input = self.df_val[self.inpt_vars].values
        self.full_val_input = self.df_val[self.inpt_vars+self.miss_vars].values
        self.known_test_input = self.df_test[self.inpt_vars].values
        self.full_test_input = self.df_test[self.inpt_vars+self.miss_vars].values
        
        if self.miss_vars:
            df_t_h = self.df_train_hypothesis
            self.inc_partial_input = df_t_h[(df_t_h["partial_full_info"]==1) & (df_t_h["correct_hypothesis"]==False)]\
                                            [self.inpt_vars + [miss_var+ '_hypothesis' for miss_var in self.miss_vars]].values
            self.latest_partial_grads = []
            self.latest_inc_partial_grads = []

            self.num_hyp_comb = self._get_num_hyp_comb()
            self._set_partial_hyp_class()
            self._check_partial_hypclass_coverage()
            
        self.batch_size = 0    
        self._prep_inpt_out_tensors()
        torch.manual_seed(self.rand_state)
        
        self.problem_type = determine_problem_type(self.df, self.target_vars[0])
        if self.miss_vars:
            self.hyp_blacklisted = self.df_train_hypothesis[(self.df_train_hypothesis["partial_full_info"]==1) & (self.df_train_hypothesis["correct_hypothesis"]==False)]["global_id"].values
        
    def _load_prep_df(self):
        self.df = self._can_load_to_df()
        if self.process_data:
            self.df = self.df.sample(frac=1, random_state = self.rand_state)
            self._remove_const_cols()
            #self._remove_const_rows()
            self._normalize_scale()
            str_cols = self._check_string_cols()
            if str_cols:
                self.has_ohe = True
                self.df, encoded_cols = one_hot_encode(self.df, str_cols)
                self.inpt_vars += encoded_cols
                self.inpt_vars = list(filter(lambda item: item not in str_cols, self.inpt_vars))
                
            else:
                self.has_ohe = False
            
    def _remove_const_cols(self):
        constant_cols = [column for column in self.df.columns if len(self.df[column].value_counts()) == 1]
        self.df.drop(columns = constant_cols, inplace = True)
        
    def _remove_const_rows(self):
        #change to avg
        self.df.drop_duplicates(subset = self.inpt_vars + self.miss_vars, keep='first', inplace = True)
        
    def _normalize_scale(self):
        has_object = False
        for col in self.df.columns:
            if (col in self.inpt_vars) or (col in self.miss_vars) or (col in self.target_vars):
                if self.df[col].dtype != 'object' and ((np.mean(self.df[col]) >= 3) or (np.std(self.df[col]) >= 3)):
                    if col in self.miss_vars:
                        #print("found col")
                        #print(self.miss_vars.index(col))
                        #print((np.min(self.df[col].values)))
                        #print((np.max(self.df[col].values) - np.min(self.df[col].values)))
                        #print([(v - np.min(self.df[col].values)) / (np.max(self.df[col].values) - np.min(self.df[col].values))
                        #                                             for v in self.hypothesis[self.miss_vars.index(col)]])
                        self.hypothesis[self.miss_vars.index(col)] = [(v - np.min(self.df[col].values)) / (np.max(self.df[col].values) - np.min(self.df[col].values))
                                                                     for v in self.hypothesis[self.miss_vars.index(col)]]
                    
                    self.df[col] = (self.df[col].values - np.min(self.df[col].values)) / (np.max(self.df[col].values) - np.min(self.df[col].values))
                    if self.verbose:
                        print(f"The column {col} was scaled. To turn off data processing set process_data to False.")
                    has_object = True

        if has_object and self.verbose:
            print("------------")
        
    def _can_load_to_df(self):
        def check_remove_unnamed_index(df):
            if "Unnamed: 0" in df.columns:
                df.drop(columns=["Unnamed: 0"], inplace = True)
            return df
       
        if os.path.exists(self.path):
            if self.path.split(".")[-1] == "csv":
                return check_remove_unnamed_index(pd.read_csv(self.path))
            elif self.path.split(".")[-1] == "tsv":
                return check_remove_unnamed_index(pd.read_csv(self.path, sep='\t'))
            else:
                raise FileNotFoundError(f"{self.path.split('.')[-1]} is not supported.")      
        else:
            raise FileNotFoundError(f"{self.path} does not exist.")

    def _check_string_cols(self):
        str_cols = []
        for col in self.inpt_vars:
            if self.df[col].dtype == 'object':
                if self.df[col].apply(lambda x: isinstance(x, str)).any():
                    str_cols.append(col)
        if str_cols and self.verbose:
            print(f"These columns: {str_cols}, contain strings and will be one hot encoded.")
        return str_cols

    def _set_hyp_class_threshold(self, hyp_class_thresh):
        if not hyp_class_thresh:
            #print("--")
            #print(self.hypothesis)
            for l_i in range(len(self.hypothesis)):
                hyp_c_l = []
                for i in range(len(self.hypothesis[l_i])-1):
                    hyp_c_l.append(np.mean([self.hypothesis[l_i][i],self.hypothesis[l_i][i+1]]))
                hyp_class_thresh.append(hyp_c_l)
        return hyp_class_thresh

    def _get_num_hyp_comb(self):
        if len(self.hypothesis) == 1:
            return len(self.hypothesis[0])
        else:
            return num_combinations(self.hypothesis)
    
    def _prep_inpt_out_tensors(self):
        self.known_val_input_tensor = torch.tensor(self.known_val_input, dtype=torch.float32).to(self.device)
        self.full_val_input_tensor = torch.tensor(self.full_val_input, dtype=torch.float32).to(self.device)
        self.known_test_input_tensor = torch.tensor(self.known_test_input, dtype=torch.float32).to(self.device)
        self.full_test_input_tensor = torch.tensor(self.full_test_input, dtype=torch.float32).to(self.device)
        self.partial_input_tensor = torch.tensor(self.partial_input, dtype=torch.float32).to(self.device)
        
        print("##########-------#############")
        print(self.partial_input_tensor.shape)
        
        if self.miss_vars:
            self.inc_partial_input_tensor = torch.tensor(self.inc_partial_input, dtype=torch.float32).to(self.device)
        
        self.val_outcomes_tensor = torch.tensor(self.df_val[self.target_vars].values, 
                                                dtype=torch.float32).to(self.device)
        self.val_outcomes_tensor = self.val_outcomes_tensor.view(-1,1)
        self.test_outcomes_tensor = torch.tensor(self.df_test[self.target_vars].values, 
                                                dtype=torch.float32).to(self.device)
        self.test_outcomes_tensor = self.test_outcomes_tensor.view(-1,1)
        
        self.partial_full_outcomes = torch.tensor(self.df_train[self.target_vars].iloc[self.partial_rows_id].values, 
                                                  dtype=torch.float32).to(self.device)
        self.partial_full_outcomes = self.partial_full_outcomes.view(-1,1)
        
    def prep_dataloader(self, use_info, batch_size, **kwargs):
        
        self.batch_size = batch_size
        
        if use_info == "full info":
            dataset = prep_tensordataset(self.full_input, self.df_train[self.target_vars].values)
        elif use_info == "use hypothesis":
            miss_vars_hypothesis = [name+"_hypothesis" for name in self.miss_vars]
            
            print("##########-------#############")
            print(self.inpt_vars)
            print(miss_vars_hypothesis)
            
            h_input = self.df_train_hypothesis[self.inpt_vars+miss_vars_hypothesis].values
            h_output = self.df_train_hypothesis[self.target_vars].values
            
            print(h_input.shape)
            #print(h_output.shape)
            
            dataset = prep_tensordataset(h_input, h_output)
        elif use_info == "use known only":
            dataset = prep_tensordataset(self.known_input, self.df_train[self.target_vars].values)
        elif use_info == "partial info":    
            dataset = prep_tensordataset(self.partial_input, 
                                         self.df_train[self.target_vars].iloc[self.partial_rows_id].values)
        elif use_info == "use imputation":
            if "imputed_input" not in kwargs:
                dataset = prep_tensordataset(self.full_input, self.df_train[self.target_vars].values)
            else:
                dataset = prep_tensordataset(kwargs["imputed_input"], self.df_train[self.target_vars].values)
       
        elif use_info == "known info noisy simulation" or use_info == "full info noisy":
            
            dataset = prep_tensordataset(self.known_input, self.df_train_noisy[self.target_vars[0]+"_noisy"].values)
        #elif use_info == "rand hyp + partial":

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    
    def get_validation_tensors(self, use_info):
        val_outcomes = torch.tensor(self.df_val[self.target_vars].values, dtype=torch.float32).to(self.device)
        val_outcomes = val_outcomes.view(-1,1)
        if use_info in ["full info", "partial info", "use hypothesis", "use imputation"]:
            return self.full_val_input_tensor, val_outcomes
        elif use_info == "use known only":
            return self.known_val_input_tensor, val_outcomes
        
    def get_test_tensors(self, use_info):
        test_outcomes = torch.tensor(self.df_test[self.target_vars].values, dtype=torch.float32).to(self.device)
        test_outcomes = test_outcomes.view(-1,1)
        if use_info in ["full info", "partial info", "use hypothesis", "use imputation"]:
            return self.full_test_input_tensor, test_outcomes
        elif use_info == "use known only":
            return self.known_test_input_tensor, test_outcomes
        
    def _set_partial_hyp_class(self):
        
        df_t_h = self.df_train_hypothesis
        self.true_partial_hyp_class = list(df_t_h[(df_t_h["partial_full_info"]==1) & 
                                                  (df_t_h["correct_hypothesis"]==True)]["hyp_class_id"].values)
        self.inc_partial_hyp_class = list(df_t_h[(df_t_h["partial_full_info"]==1) & 
                                                  (df_t_h["correct_hypothesis"]==False)]["hyp_class_id"].values)
    
    def append2hyp_df(self, batch_id, array, column, layer = -2):
        """
        Appends values from a 1D array to a list in each row of a specific column in a pandas dataframe.
        
        Parameters:
        df (pd.DataFrame): The input dataframe.
        column (str): The specific column in the dataframe.
        array (list): The 1D array to take values from.
        batch_size (int): The number of rows in each batch.
        batch_id (int): The batch number to start appending from.

        """

        start_index = batch_id * self.batch_size
        #end_index = start_index + self.batch_size
    
        if column == "gradients":
            if self.device == "cpu":
                array = [grad[layer].numpy().astype('float16') for grad in array]
            else:
                array = [grad[layer].cpu().numpy().astype('float16') for grad in array]
            
        for i, val in enumerate(array):
            if self.use_case == "hypothesis":
                self.df_train_hypothesis.at[i+start_index, column].append(val)
            elif self.use_case == "noise detection":
                self.df_train_noisy.at[i+start_index, column].append(val)
            
        # # Check if the column exists, if not create it
        # if column not in self.df_train_hypothesis.columns:
        #     self.df_train_hypothesis[column] = [list() for _ in range(len(self.df_train_hypothesis))]
    
        # # Append values from array to the specific column in the dataframe
        # for i in range(start_index, min(end_index, self.df_train_hypothesis.shape[0])):
        #     if (i-start_index) < len(array): # Check if index is within array length
        #         self.df_train_hypothesis.at[i, column].append(array[i-start_index])
        #     else: # If array is smaller than batch_size, fill remaining with None
        #         self.df_train_hypothesis.at[i, column].append(None)

    def _create_hypothesis_data(self):
        
        df_train = self.df_train.copy()
        df_train = df_train[self.inpt_vars + self.miss_vars + self.target_vars]
        df_train.loc[:,"partial_full_info"] = 0
        df_train.reset_index(inplace = True, drop = True)
        df_train.loc[self.partial_rows_id, "partial_full_info"] = 1
        df_train["correct_class_id"] = get_hyp_classes(self.full_input, self.hyp_class_thresh[0])
        if self.miss_vars:
            self.df_train_hypothesis = expand_df_hypothesis(df_train, self.inpt_vars, self.hypothesis[0], self.miss_vars[0])
            self.df_train_hypothesis["correct_hypothesis"] = self.df_train_hypothesis.apply(lambda row: 
                                                                            row["correct_class_id"]==row["hyp_class_id"],
                                                                            axis = 1)
            self.df_train_hypothesis = self._add_tracking_cols(self.df_train_hypothesis)
        
    def _add_tracking_cols(self, df):
        df['global_id'] = [i for i in range(len(df))]          
        df["sel_hyp_tracker"] = [[] for v in range(len(df))]
        df["final_sel_hyp"] = [[] for v in range(len(df))]
        df["loss"] = [[] for v in range(len(df))]
        df["gradients"] = [[] for v in range(len(df))]
        return df
        
    def _check_partial_hypclass_coverage(self):
        self.lack_partial_coverage = False
        for v in self.df_train_hypothesis["correct_class_id"].value_counts().index:
            if v not in self.df_train_hypothesis[self.df_train_hypothesis["partial_full_info"]== 1]["correct_class_id"].unique():
                self.lack_partial_coverage = True
                
    def simulate_noise(self, DATA_NOISE_PERC, NOISE_MINRANGE, NOISE_MAXRANGE):
        
        self.df_train_noisy = deepcopy(self.df_train[self.inpt_vars+self.target_vars])
        self.df_train_noisy.reset_index(inplace = True, drop = True)
        self.noise_rows_id = self._sel_rows_to_add_noise(DATA_NOISE_PERC)
        self._add_noise_to_target(NOISE_MINRANGE, NOISE_MAXRANGE)
        
        self.df_train_noisy = self._add_tracking_cols(self.df_train_noisy)
        
    def _sel_rows_to_add_noise(self, DATA_NOISE_PERC):
        random.seed(self.rand_state)
        return random.sample(range(0, len(self.df_train_noisy)-1), int(len(self.df_train_noisy)*DATA_NOISE_PERC))

    def _add_noise_to_target(self, NOISE_MINRANGE, NOISE_MAXRANGE):
        y_range = self.df_train_noisy[self.target_vars[0]].max() - self.df_train_noisy[self.target_vars[0]].min()

        self.df_train_noisy["noise_added"] = 0
        self.df_train_noisy[self.target_vars[0]+"_noisy"] = deepcopy(self.df_train_noisy[self.target_vars[0]].values)
        for row_id in self.noise_rows_id:
            self.df_train_noisy.loc[row_id, "noise_added"] = 1
            if random.randint(1, 10) > 5: 
                self.df_train_noisy.loc[row_id, self.target_vars[0]+"_noisy"] = self.df_train_noisy.loc[row_id, self.target_vars[0]] + random.uniform(NOISE_MINRANGE*y_range,NOISE_MAXRANGE*y_range)
            else:
                self.df_train_noisy.loc[row_id, self.target_vars[0]+"_noisy"] = self.df_train_noisy.loc[row_id, self.target_vars[0]] - random.uniform(NOISE_MINRANGE*y_range,NOISE_MAXRANGE*y_range)

    
        
def one_hot_encode(df, columns_to_encode, ohe_weigth = 1.0):
    """
    Convert specified columns of a pandas DataFrame into one-hot encoded columns.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to one-hot encode.
    columns_to_encode : list of str
        A list of column names to one-hot encode.

    Returns
    -------
    pandas DataFrame
        A new DataFrame with the specified columns one-hot encoded.
    """
    # Create a copy of the original DataFrame to avoid modifying it directly
    df_encoded = df.copy()
    encoded_columns = []
    # Loop through each column to encode
    for column in columns_to_encode:
        # One-hot encode the column using pandas get_dummies function
        one_hot_encoded = pd.get_dummies(df[column], prefix=column)
        one_hot_encoded = one_hot_encoded.replace(1, 1*ohe_weigth)
        encoded_columns.extend(one_hot_encoded.columns)
        # Add the one-hot encoded columns to the new DataFrame
        df_encoded = pd.concat([df_encoded, one_hot_encoded], axis=1)

    # Drop the original columns from the new DataFrame
    df_encoded.drop(columns=columns_to_encode, inplace=True)

    return df_encoded, encoded_columns

def num_combinations(lists: List[List[int]]) -> int:
    """
    Calculates the number of possible combinations that can be made by using one element from each list
    """
    num_combinations = 1
    for lst in lists:
        num_combinations *= len(lst)
    return num_combinations

def get_hyp_classes(input_vectrs, hyp_thresholds):
    hyp_classes = []
    for vec in input_vectrs:
        for i, threshold in enumerate(hyp_thresholds):
            #print(type(input_vectrs[0]))
            #print(input_vectrs[0])
            #print(input_vectrs[0][-1])
            #print(threshold)
            if vec[-1] <= threshold:
                hyp_classes.append(i)
                break
        else:
            # If row[-1] doesn't meet any threshold, append the length of thresholds
            hyp_classes.append(len(hyp_thresholds))
    return hyp_classes

def expand_df_hypothesis(df, inpt_vars, hypothesis, miss_var):
    # Create a list to store each expanded section of the DataFrame
    dfs = []
    
    # The original index will help us to replicate each row the same number of times
    # as in the original function
    original_index = df.index
    
    for hyp_class_id, value in enumerate(hypothesis):
        # Duplicate each row for the current hypothesis value
        df_temp = df.loc[original_index.repeat(len(hypothesis))].reset_index(drop=True)
        
        # Only keep rows that correspond to the current hypothesis
        mask = df_temp.index % len(hypothesis) == hyp_class_id
        df_temp = df_temp[mask]
        
        # Add new columns for the current hypothesis value and hyp_class_id
        df_temp[miss_var + '_hypothesis'] = value
        df_temp['hyp_class_id'] = hyp_class_id
        
        dfs.append(df_temp)
    
    # Concatenate all the DataFrames
    df_result = pd.concat(dfs).sort_index(kind='merge')
    
    # Reset the index of the result dataframe
    df_result.reset_index(drop=True, inplace=True)
    
    # Reorder columns
    cols = df_result.columns.tolist()
    cols = cols[:cols.index(inpt_vars[-1])+1] + [cols[-1]] + cols[cols.index(inpt_vars[-1])+1:-1]
    df_result = df_result[cols]
    
    return df_result

def prep_tensordataset(input_array, targets):
    return TensorDataset(torch.tensor(input_array, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))

def flatten_list(lst):
    """
    Recursively flattens a list of lists.
    """
    if isinstance(lst, list):
        flattened = []
        for elem in lst:
            flattened.extend(flatten_list(elem))
        return flattened
    else:
        return [lst]
    
def duplicate_elements(tensor, X):
    # Repeat the tensor along a new dimension X times
    repeated_tensor = tensor.repeat_interleave(X)

    return repeated_tensor

def remove_values(input_tensor, indx_to_remove):
    """
    Removes values from the input tensor at the given indices.
    """
    indx_to_remove = sorted(indx_to_remove, reverse=True)
    for idx in indx_to_remove:
        input_tensor = torch.cat([input_tensor[:idx], input_tensor[idx+1:]])

    return input_tensor


def remove_binary_columns(input_tensor, indx_to_remove=None):
    if input_tensor.dim() == 1 and indx_to_remove is not None:
        # Remove specified elements by index
        output_tensor = remove_values(input_tensor, indx_to_remove)
        removed_indxs = indx_to_remove
    else:
        if indx_to_remove is not None:
            # Remove specified columns by index
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            output_tensor = torch.cat((input_tensor[:, :indx_to_remove[0]], input_tensor[:, indx_to_remove[0]+1:]), dim=1)
            output_tensor = output_tensor.squeeze(0)  # Remove batch dimension
            removed_indxs = indx_to_remove
        else:
            # Find columns that are not composed of only 0s and 1s
            non_binary_columns = ((input_tensor != 0) & (input_tensor != 1)).any(dim=0)

            # Select only columns that are not binary
            output_tensor = input_tensor[:, non_binary_columns]
            removed_indxs = torch.where(non_binary_columns == False)[0].tolist()

    return output_tensor, removed_indxs

def determine_problem_type(df, target_column):
    """
    Determines the type of machine learning problem based on a dataframe and a target column.
    
    Parameters:
    - df: DataFrame, the input data.
    - target_column: str, the name of the target column.
    
    Returns:
    - str, the type of the machine learning problem: 'Regression', 'Binary Classification', or 'Multi-class Classification'.
    """
    
    # Check if the target column exists in the dataframe
    if target_column not in df.columns:
        return "Target column not found in the DataFrame."
    
    # Extract the unique values in the target column
    unique_values = df[target_column].unique()
    
    # Check if the target column has only numerical values
    if pd.api.types.is_numeric_dtype(df[target_column]):
        
        # If there are many unique values, it's likely a regression problem
        if len(unique_values) > 10:  
            return 'regression'
        
        # If there are only 2 unique values, it's a binary classification problem
        elif len(unique_values) == 2:
            return "binary-class"
        
        # Otherwise, it's likely a multi-class classification problem
        else:
            return "multi-class"
        
    # If the target column has non-numerical values
    else:
        if len(unique_values) == 2:
            return "binary-class"
        else:
            return "multi-class"       
        

















