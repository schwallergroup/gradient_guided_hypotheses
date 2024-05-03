import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.svm import OneClassSVM

from sklearn.cluster import KMeans, DBSCAN

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from .data_ops import remove_binary_columns

class AlgoModulators():
    def __init__(self, data_operator, lr = 0.001, dropout = 0.05, nu = 0.1, normalize_grads_contx = False, use_context = True, eps_value = "-", min_samples_ratio = "-"):
        self.epoch_loss_in_contxt = 4
        self.sel_freq_crit_start_after = 4
        self.partial_freq_per_epoch = 2
        self.freqperc_cutoff = 0.33
        self.bi_class_sel = False
        self.gradwcontext = use_context  #True if IPv3
        self.cluster_all_classes = False
        self.has_ohe = False
        self.save_results = False
        self.no_selection_epochs = 0
        self.rmv_avg_grad_signal = True
        self.layer = -2
        
        self.select_low_loss_cluster = False
        self.n_clusters = 5
        
        self._get_data_specs(data_operator)
        self._mod_algo_for_data()
        
        self.lr = lr
        self.dropout = dropout
        self.nu = nu
        self.normalize_grads_contx = normalize_grads_contx
        #torch.manual_seed(42)
        
        self.eps_value = eps_value
        self.min_samples_ratio = min_samples_ratio
        
        
    def _get_data_specs(self, data_operator):
        self.has_ohe = data_operator.has_ohe
        
    def _mod_algo_for_data(self):
        if self.has_ohe:
            self.freqperc_cutoff = 0.30       
            
from torch.nn.parallel import DataParallel
from torch.autograd import grad
import concurrent.futures
import multiprocessing
import os
import torch.nn as nn

def compute_individual_grads(model, individual_losses, device):
    # Move model and losses to GPU if available
    if device == "gpu":
        individual_losses = [loss.to(device) for loss in individual_losses]
    
    # Create a thread pool with as many worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()-2) as executor:
        # Submit the grad function calls to the executor as tasks
        individual_grads = [executor.submit(grad, individual_loss, model.parameters(), retain_graph=True) for individual_loss in individual_losses]
        concurrent.futures.wait(individual_grads)
    
    individual_grads = [future.result() for future in individual_grads]
                
    return individual_grads

import random

def gradient_random_selection(hypothesys_grads, HYPOTHESIS_COMB_LENGHT):
    # Initialize an empty list to store the selected elements
    selected_elements = []
    # Iterate over the list by block size
    for i in range(0, len(hypothesys_grads), HYPOTHESIS_COMB_LENGHT):
        # Get the current block
        block = hypothesys_grads[i:i+HYPOTHESIS_COMB_LENGHT]
        # Select a random element from the block
        selected_element = random.choice(block)
        # Append the selected element to the list of selected elements
        selected_elements.append(selected_element)
    return selected_elements

class MSEIndividualLosses(nn.MSELoss):
    def forward(self, predictions, labels):

        individual_losses = (predictions - labels) ** 2
        #print(individual_losses.size())
        overall_loss = individual_losses.mean()
        return overall_loss, individual_losses
    
class BCEIndividualLosses(nn.BCELoss):
    def forward(self, predictions, labels):
        # Apply sigmoid activation to convert predictions to probabilities
        predictions = torch.sigmoid(predictions)
        
        # Calculate individual binary cross-entropy losses
        individual_losses = - (labels * torch.log(predictions) + (1 - labels) * torch.log(1 - predictions))
        
        # Calculate the overall BCE loss
        overall_loss = individual_losses.mean()
        
        return overall_loss, individual_losses
    
class CrossEntropyIndividualLosses(nn.CrossEntropyLoss):
    def forward(self, predictions, labels):
        # Calculate individual cross-entropy losses
        individual_losses = -torch.log(predictions[range(len(labels)), labels])
        
        # Calculate the overall cross-entropy loss
        overall_loss = individual_losses.mean()
        
        return overall_loss, individual_losses

def train_model(X_train, y_train, model_name='logistic_regression'):
    if model_name == 'logistic_regression':
        model = LogisticRegression(class_weight='balanced')
    elif model_name == 'random_forest':
        model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    elif model_name == 'svm':
        model = SVC(class_weight='balanced', kernel='rbf', C=1.0, gamma='scale')
    else:
        raise ValueError(f"Invalid model name: {model_name}. Supported models: 'logistic_regression', 'random_forest', 'svm'.")

    model.fit(X_train, y_train)
    return model

def check_grads_blacklisted(full_grads, inc_full_grads):
    for inc_full_grad in inc_full_grads:
        if torch.equal(full_grads[-2], inc_full_grad[-2]):
            #print("avoided")
            return True
    return False

def check_hyp_blacklisted(DO, global_hyp_id):
    if global_hyp_id in DO.df_train_hypothesis[(DO.df_train_hypothesis["partial_full_info"]==1) & (DO.df_train_hypothesis["correct_hypothesis"]==False)]["global_id"].values:
        return True
    else:
        return False
        

def find_global_hyp_id(batch_i, batch_size, hyp_per_sample, class_id, local_id):
    if class_id == "all classes":
        return batch_i*batch_size*hyp_per_sample+local_id
    else:
        return batch_i*batch_size+local_id*hyp_per_sample+class_id  #*hyp_per_sample

def gradient_selection(DO, AM, epoch, hypothesis_grads, partial_full_grads, batch_size, 
                       hyp_per_sample, batch_i, input_hypothesis, partial_full_input,
                       labels, predictions, partial_full_outcomes, partil_full_predictions,
                       individual_losses, partial_individual_losses, incorr_partial_inpts, partial_incorr_full_preds,
                       inc_partial_individual_losses, inc_partial_full_grads, rand_state):    

    #print(partial_full_input)
    all_hyp = [[] for i in range(hyp_per_sample)]
    all_full_hyp = [[] for i in range(hyp_per_sample)]
    all_partial_hyp = [[] for i in range(hyp_per_sample)]
    all_inc_partial_hyp = [[] for i in range(hyp_per_sample)]
    
    labels = labels.clone().detach().requires_grad_(False)
    partial_full_outcomes = partial_full_outcomes.clone().detach().requires_grad_(False)
    #predictions = predictions.clone().detach().requires_grad_(False) 
    #partil_full_predictions = partil_full_predictions.clone().detach().requires_grad_(False)
    
    individual_losses = individual_losses.clone().detach().requires_grad_(False)
    partial_individual_losses = partial_individual_losses.clone().detach().requires_grad_(False)
    if AM.bi_class_sel:
        inc_partial_individual_losses = inc_partial_individual_losses.clone().detach().requires_grad_(False)
    
    #remove max(10) leave len(hypothesis_grads) // hyp_per_sample
    for group in range(0, max(10, len(hypothesis_grads) // hyp_per_sample)):
        group_grads = hypothesis_grads[group*hyp_per_sample:(group+1)*hyp_per_sample]
        group_inpt_hyp = input_hypothesis[group*hyp_per_sample:(group+1)*hyp_per_sample]
        #group_outpt_hyp = labels[group*hyp_per_sample:(group+1)*hyp_per_sample] #predictions
        
        if epoch > AM.epoch_loss_in_contxt:
            group_loss_hyp = individual_losses[group*hyp_per_sample:(group+1)*hyp_per_sample]
        grad_interest = []
        for i, grads in enumerate(group_grads):
            for j, gradi in enumerate(grads[-2:-1]):
                grad_interest.append(gradi)          
        #group_grad_mean = torch.mean(torch.stack(grad_interest), dim=0)
        #grad_interest = [gradi-group_grad_mean for gradi in grad_interest]
        grad_interest = [vector.view(-1,1) for vector in grad_interest]
        grad_interest = torch.cat(grad_interest, dim = 1)
        
        if AM.gradwcontext == True:
            if AM.has_ohe:
                pass
                #group_inpt_hyp, removed_indxs = remove_binary_columns(group_inpt_hyp)
            group_inpt_hyp = torch.transpose(group_inpt_hyp, 0, 1)
            #group_outpt_hyp = torch.transpose(group_outpt_hyp, 0, 1)
            
            if epoch > AM.epoch_loss_in_contxt:
                group_loss_hyp = torch.transpose(group_loss_hyp, 0, 1)
                grad_context_interest = torch.cat([grad_interest, group_inpt_hyp, group_loss_hyp], dim = 0) # group_outpt_hyp,
            else:
                grad_context_interest = torch.cat([grad_interest, group_inpt_hyp], dim = 0) #, group_outpt_hyp
            [all_hyp[i].append(grad_context_interest[:,i]) for i in range(hyp_per_sample)]
        else:
            [all_hyp[i].append(grad_interest[:,i]) for i in range(hyp_per_sample)]

    for i in range(0,len(hypothesis_grads), hyp_per_sample):
        [all_full_hyp[j].append(hypothesis_grads[i+j]) for j in range(hyp_per_sample)]
    
    for grad_int, part_inpt, part_loss, part_outpt, t_h in zip(partial_full_grads, partial_full_input, partial_individual_losses,
                                                               partial_full_outcomes, DO.true_partial_hyp_class): #partil_full_predictions,
        if AM.gradwcontext:
            if AM.has_ohe:
                pass
                #part_inpt, _ = remove_binary_columns(part_inpt, removed_indxs)
            grad_int = grad_int[-2].reshape(grad_int[-2].shape[1])
            if epoch > AM.epoch_loss_in_contxt:
                partial_grad_contxt = torch.cat([grad_int, part_inpt, part_loss], dim = 0) #, part_outpt  part_inpt/100,
            else:
                partial_grad_contxt = torch.cat([grad_int, part_inpt], dim = 0) #, part_outpt, part_inpt/100,
            # print(t_h)
            # print(partial_grad_contxt)
            # import sys
            # sys.exit()
            all_partial_hyp[t_h].append(partial_grad_contxt)
        else:
            all_partial_hyp[t_h].append(grad_int[-2].reshape(grad_int[-2].shape[1]))
        
#    if AM.bi_class_sel:
#        for grad_int, part_inpt, part_loss, part_outpt, t_h in zip(inc_partial_full_grads, incorr_partial_inpts, inc_partial_individual_losses,
#                                                                   partial_incorr_full_preds, DO.inc_partial_hyp_class):
#            if AM.gradwcontext:
#                if AM.has_ohe:
#                    pass
#                    #part_inpt, _ = remove_binary_columns(part_inpt, removed_indxs)
#                grad_int = grad_int[-2].reshape(grad_int[-2].shape[1])
#                if epoch > AM.epoch_loss_in_contxt:
#                    partial_grad_contxt = torch.cat([grad_int, part_inpt, part_loss], dim = 0) #, part_outpt  part_inpt/100,
#                else:
#                    partial_grad_contxt = torch.cat([grad_int, part_inpt], dim = 0) #, part_outpt, part_inpt/100,
#                all_inc_partial_hyp[t_h].append(partial_grad_contxt)
#            else:
#                all_inc_partial_hyp[t_h].append(grad_int[-2].reshape(grad_int[-2].shape[1]))                
    
        
    selected_gradients = []
    selected_global_ids = []
    

    if AM.bi_class_sel:
        use_model = 'logistic_regression' #logistic_regression; random_forest, 'svm' 0.0095 epochs 100 per class; logistic_regression 0.0085 epochs 45 ;random_forest 0.012 epochs 50; 
        for class_id, (hyp_class_grads, hyp_full_class_grads, hyp_partial_class_grads, hyp_inc_partial_class_grads) in enumerate(zip(all_hyp, all_full_hyp, all_partial_hyp, all_inc_partial_hyp)):
            class_1_vectors = [gradi.numpy() for gradi in hyp_partial_class_grads]
            class_2_vectors = [gradi.numpy() for gradi in hyp_inc_partial_class_grads]
            unknown_vectors = [gradi.numpy() for gradi in hyp_class_grads]
            
            X_train, y_train = prepare_binary_classification_data(class_2_vectors, class_1_vectors, random_state=DO.rand_state)
            grad_bi_model = train_model(X_train, y_train, model_name=use_model)

            unknown_labels = list(grad_bi_model.predict(unknown_vectors)) 
            
            for local_id, (u_l, full_grads) in enumerate(zip(unknown_labels, hyp_full_class_grads)):
                if u_l == 1:
                    global_hyp_id = find_global_hyp_id(batch_i, batch_size, hyp_per_sample, class_id, local_id)
                    selected_global_ids.append(global_hyp_id)                  
                    DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(1)
                    
                    #global_id_str = str(find_global_hyp_id(batch_i, batch_size, hyp_per_sample, class_id, local_id))
                    if epoch > AM.sel_freq_crit_start_after:
                        if sum(DO.df_train_hypothesis.sel_hyp_tracker.iloc[global_hyp_id]) >= epoch*AM.freqperc_cutoff: 
                            selected_gradients.append(full_grads)
                    else:
                        selected_gradients.append(full_grads) 
    else:
        for class_id, (hyp_class_grads, hyp_full_class_grads, hyp_partial_class_grads) in enumerate(zip(all_hyp, all_full_hyp, all_partial_hyp)):
            #print(len(hyp_partial_class_grads))
            if DO.device == "cpu":
                class_1_vectors = [gradi.numpy() for gradi in hyp_partial_class_grads]
                unknown_vectors = [gradi.numpy() for gradi in hyp_class_grads]
            else:
                class_1_vectors = [gradi.cpu().numpy() for gradi in hyp_partial_class_grads]
                unknown_vectors = [gradi.cpu().numpy() for gradi in hyp_class_grads]
            
            #! Experiment with reducing feature importance weight from non gradients sources when OHE is true
            if AM.normalize_grads_contx:
                _, class_1_vectors, unknown_vectors = normalize_and_split(np.asarray(class_1_vectors), np.asarray(unknown_vectors))
            
            #feature_weights = np.array([1 for i in range(len(grad_int))]+[0.60 for i in range(class_1_vectors.shape[1]-len(grad_int))])  # Adjust these weights as needed
            #print(class_1_vectors)
            #class_1_vectors = class_1_vectors * feature_weights
            #print(class_1_vectors)
            #unknown_vectors = unknown_vectors * feature_weights
            
            #increase in nu will make clusters more restrictive move selection distributions to the left
            grad_model = OneClassSVM(kernel='poly', nu = AM.nu).fit(class_1_vectors) #1/hyp_per_sample
            #grad_model = IsolationForest(contamination=0.5).fit(class_1_vectors)
            #grad_model= EllipticEnvelope(contamination=0.2).fit(class_1_vectors)
            
            unknown_labels = list(grad_model.predict(unknown_vectors))
            #print(sum(unknown_labels))
            
            #Experiment to select lowest loss cluster
#            if AM.select_low_loss_cluster:
#                unknown_labels = select_low_loss_cluster(unknown_vectors, AM.n_clusters, rand_state)
                #print(unknown_labels[:10])
            #print(unknown_labels)

#            for local_id, (u_l, full_grads) in enumerate(zip(unknown_labels, hyp_full_class_grads)):
#                #hm.append(u_l)
#                global_hyp_id = find_global_hyp_id(batch_i, batch_size, hyp_per_sample, class_id, local_id) # batch_size*batch_i+local_id  #
#                if u_l == 1:
##                    if not check_grads_blacklisted(full_grads, inc_partial_full_grads):
#                        selected_global_ids.append(global_hyp_id)
#                        DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(1)
#                        #global_id_str = str(find_global_hyp_id(batch_i, batch_size, hyp_per_sample, class_id, local_id))
#                        if epoch > AM.sel_freq_crit_start_after:
#                            if sum(DO.df_train_hypothesis.sel_hyp_tracker.iloc[global_hyp_id]) >= epoch/3:
#                                selected_gradients.append(full_grads)
#                                
#                                DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(1)
#                            else:
#                                DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(0)
#                        else:
#                            selected_gradients.append(full_grads)
#                            DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(1)
#                    else:
#                        DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(0)
#                        DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(0)
#                else:
#                    DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(0)
#                    DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(0)  
                    
            for local_id, (u_l, full_grads) in enumerate(zip(unknown_labels, hyp_full_class_grads)):
                #hm.append(u_l)
                global_hyp_id = find_global_hyp_id(batch_i, batch_size, hyp_per_sample, class_id, local_id) # batch_size*batch_i+local_id  #
                if epoch > AM.no_selection_epochs:
                    if u_l == 1:
                        if global_hyp_id not in DO.hyp_blacklisted:# check_hyp_blacklisted(DO, global_hyp_id):   #check_grads_blacklisted(full_grads, inc_partial_full_grads):
                            selected_global_ids.append(global_hyp_id)
                            DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(1)
                            #global_id_str = str(find_global_hyp_id(batch_i, batch_size, hyp_per_sample, class_id, local_id))
                            if epoch > AM.sel_freq_crit_start_after:
                                if sum(DO.df_train_hypothesis.sel_hyp_tracker.iloc[global_hyp_id]) >= epoch*AM.freqperc_cutoff:
                                    selected_gradients.append(full_grads)

                                    DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(1)
                                else:
                                    DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(0)
                            else:
                                selected_gradients.append(full_grads)
                                DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(1)
                        else:
                            DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(0)
                            DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(0)
                    else:
                        DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(0)
                        DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(0)
                else:
                    selected_gradients.append(full_grads)
                    DO.df_train_hypothesis.at[global_hyp_id, 'sel_hyp_tracker'].append(1)
                    DO.df_train_hypothesis.at[global_hyp_id, 'final_sel_hyp'].append(1)                    
            
            
            #print(selected_gradients)
            #print(len(selected_gradients))
            #import sys
            #sys.exit()
                    
    #print(hm)
    return selected_gradients, selected_global_ids
    

def select_low_loss_cluster(unknown_vectors, n_clusters, rand_state):
    """
    Labels tensors based on k-means clustering and the average of their last values.
    
    Parameters:
    - unknown_vectors: list of tensors
    - n_clusters: number of clusters for k-means
    
    Returns:
    - labels: list of labels (0 or 1)
    """
    
    # Apply k-means clustering
    # Assuming unknown_vectors are 2D. If they are not, you might need to reshape them.
    flattened_tensors = [tensor.flatten() for tensor in unknown_vectors]
    kmeans = KMeans(n_clusters=n_clusters, random_state=rand_state, n_init=10).fit(flattened_tensors)
    
    # Calculate the average of the last value for each cluster
    averages = []
    for i in range(n_clusters):
        avg = np.mean([tensor[-1] for idx, tensor in enumerate(unknown_vectors) if kmeans.labels_[idx] == i])
        averages.append(avg)
    
    # Find the cluster with the lowest average
    cluster_with_lowest_avg = np.argmin(averages)
    
    # Label the unknown_vectors
    labels = [1 if label == cluster_with_lowest_avg else 0 for label in kmeans.labels_]
    
    return labels


def prepare_binary_classification_data(class_0_arrays, class_1_arrays, random_state=42):
    # Combine the two lists of numpy arrays into a single dataset
    X = np.concatenate([class_0_arrays, class_1_arrays], axis=0)

    # Assign labels to each numpy array based on their class
    y = np.concatenate([np.zeros(len(class_0_arrays)), np.ones(len(class_1_arrays))])

    # Shuffle the data
    X, y = shuffle(X, y, random_state=random_state)

    # Standardize the features
    scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    return X, y
            
def hyp2hyp_binlist(hyp_class, hyp_per_sample):
    hyp_list = [0]*hyp_per_sample
    hyp_list[hyp_class] = 1
    return hyp_list
            
def gradients_mean(sel_grads):
    # Initialize the average gradients to zero
    average_grads = [torch.zeros_like(grad) for grad in sel_grads[0]]
    
    # Sum the gradients
    for grads in sel_grads:
        for i, grad in enumerate(grads):
            average_grads[i] += grad
    
    # Divide by the number of samples
    num_samples = len(sel_grads)
    for i, grad in enumerate(average_grads):
        average_grads[i] /= num_samples
    
    #different attempt
    #stkd_sel_grads = torch.stack(sel_grads, dim=0)
    #torch.mean(stkd_sel_grads, dim=0)
    return average_grads 

def normalize_and_split(arr1, arr2):
    """
    Normalize the columns of two NumPy arrays stacked vertically, and then separate them into
    two normalized arrays.

    Args:
    arr1 (numpy.ndarray): First input 2D array.
    arr2 (numpy.ndarray): Second input 2D array.

    Returns:
    numpy.ndarray: Normalized array of the combined input arrays.
    numpy.ndarray: Normalized array of the first input array.
    numpy.ndarray: Normalized array of the second input array.
    """
    combined_arr = np.vstack((arr1, arr2))

    normalized_combined = normalize(combined_arr)

    # Split the normalized array back into two separate arrays
    num_rows_arr1 = arr1.shape[0]
    normalized_arr1 = normalized_combined[:num_rows_arr1, :]
    normalized_arr2 = normalized_combined[num_rows_arr1:, :]

    return normalized_combined, normalized_arr1, normalized_arr2

def normalize(array):
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    # Avoid division by zero by handling columns with a standard deviation of 0
    std[std == 0] = 1.0
    array_normalized = (array - mean) / std
    return array_normalized

def gradient_selection_avoid_noise(AM, epoch, all_grads, batch_size, batch_i, inputs, predictions,
                                   individual_losses):
    if type(AM.min_samples_ratio) == str or type(AM.eps_value) == str:
        raise TypeError("AM noise detection parameters are not defined.")
    grad_interest = get_2nd2last_grad(all_grads, inputs, individual_losses, AM.gradwcontext)
    corr_grads, noisy_grads, corr_indx, noisy_indx = process_tensors_with_dbscan(all_grads, grad_interest, eps=AM.eps_value, min_samples=int(len(individual_losses)*AM.min_samples_ratio))

    return corr_grads, " "

def get_2nd2last_grad(all_grads, inputs, individual_losses, gradwcontext, layer = -2):
    grad_interest = []
    for i, (grads, inpt, ind_loss) in enumerate(zip(all_grads, inputs, individual_losses)):
        for j, gradi in enumerate(grads[layer:-1]):
            if gradwcontext:
                #inpt is being scaled to be inline with gradients
                inpt = torch.transpose(inpt.view(-1,1), 0, 1)/100
                ind_loss = torch.transpose(ind_loss.view(-1,1), 0, 1)
                grad_contxt = torch.cat([gradi, inpt, ind_loss], dim = 1) # ind_loss
                grad_interest.append(grad_contxt)
            else:
                grad_interest.append(gradi)
    return grad_interest

def process_tensors_with_dbscan(all_grads, tensors, eps: float = 0.2, min_samples: int = 10):
    
    # Flatten the tensors and convert the list of tensors to a NumPy array
    data = torch.stack([t.flatten() for t in tensors]).detach().numpy()
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    
    # Separate normal tensors and outliers
    normal_tensors = []
    outlier_tensors = []
    normal_indices = []
    outlier_indices = []
    
    for i, label in enumerate(clustering.labels_):
        if label == -1:  # Outliers are labeled as -1 in DBSCAN
            outlier_tensors.append(all_grads[i])
            outlier_indices.append(i)
        else:
            normal_tensors.append(all_grads[i])
            normal_indices.append(i)
    
    return normal_tensors, outlier_tensors, normal_indices, outlier_indices