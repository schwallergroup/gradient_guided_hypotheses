import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from .selection_algorithms import gradient_selection, compute_individual_grads, gradient_random_selection, \
                                 MSEIndividualLosses, BCEIndividualLosses, CrossEntropyIndividualLosses, \
                                 gradients_mean, gradient_selection_avoid_noise
from .custom_optimizer import CustomAdam
from .data_ops import duplicate_elements
from .models import initialize_model, load_model
import os

class TrainValidationManager():
    def __init__(self, use_info, num_epochs, dataloader, batch_size, rand_state, save_path, select_gradients = False, end_epochs_noise_detection = 0, best_valid_error = np.inf, imput_method = "", final_analysis = False):
        self.use_info = use_info
        self.save_path = save_path
        self.rand_state = rand_state
        self.final_analysis = final_analysis
        if not self.final_analysis:
            self.weights_save_path = f"{self.save_path}/{self.use_info}/{self.rand_state}.pt"
            self.model_save_path = f"{self.save_path}/{self.use_info}/{self.rand_state}.pth"
        else:
            if self.use_info != "use imputation":
                self.weights_save_path = f"{self.save_path}/{self.use_info}/final_analysis/{self.rand_state}.pt"
                #self.model_save_path = f"{self.save_path}/{self.use_info}/final_analysis/{self.rand_state}.pth"
            else:          
                self.weights_save_path = f"{self.save_path}/{self.use_info}/final_analysis_{imput_method}/{self.rand_state}.pt"
                if not os.path.exists(f"{self.save_path}/{self.use_info}/final_analysis_{imput_method}"):
                    os.makedirs(f"{self.save_path}/{self.use_info}/final_analysis_{imput_method}")
        
        if not os.path.exists(f"{self.save_path}/{self.use_info}"):
            os.makedirs(f"{self.save_path}/{self.use_info}")
        
        self.num_epochs = num_epochs
        self.end_epochs_noise_detection = end_epochs_noise_detection
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.train_errors_epoch = []
        self.valid_errors_epoch = []
        self.val_epochs = []
        self.best_valid_error = best_valid_error
        self.best_checkpoint = 0
        self.sel_grads_num_logs = []
        self.all_partial_ind_losses = {}
        for i in range(num_epochs):
            self.all_partial_ind_losses[f"epoch_{i}"] = []
        
        self.corr_vs_incor_loss = {}
        self.sel_corr_vs_incor_loss = {}
        
        self.select_gradients = select_gradients
        if self.use_info == "use hypothesis":
            self.select_gradients = True
  
        self.use_tqdm = True
        if self.use_info in ["full info", "use known only", "partial info", "use imputation", "full info noisy"]:
            self.use_tqdm = False   
            
        if self.use_info in ["known info noisy simulation"]:
            self.best_eps = "-"
            self.min_samples_ratio = "-"
    
    def train_model(self, DO, AM, model, final_analysis = False):
        
        #torch.manual_seed(self.rand_state)
        if final_analysis and not os.path.exists(self.save_path+f"/{self.use_info}/final_analysis") and self.use_info != "use imputation":
            os.makedirs(self.save_path+f"/{self.use_info}/final_analysis")
        
        custom_optimizer = CustomAdam(model.parameters(), lr= AM.lr) #, weight_decay=0.005
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=AM.lr) #, weight_decay=0.005
        if model.type == "regression":
            self.loss_fn = nn.MSELoss()
            loss_fn_custom = MSEIndividualLosses()
        elif model.type == "binary-class":
            self.loss_fn = nn.BCELoss()
            loss_fn_custom = BCEIndividualLosses()
        elif model.type == "multi-class":
            self.loss_fn = nn.CrossEntropyLoss()
            loss_fn_custom = CrossEntropyIndividualLosses()
        num_validations = max(int(self.num_epochs/5), 1)
        first_load = True


        for epoch in range_with_tqdm(self.num_epochs, self.use_tqdm):

            train_errors = []
            self.corr_vs_incor_loss[epoch] = [[],[]]
            self.sel_corr_vs_incor_loss[epoch] = [[],[]]
            for batch_i, (inputs, labels) in enumerate(self.dataloader):

                #print(batch_i)

                inputs = inputs.to(DO.device)
                labels = labels.to(DO.device)
                predictions = model(inputs)
                labels = labels.view(-1,1)

                if self.use_info in ["full info", "use known only", "partial info", "use imputation", "full info noisy"]:


                    loss = self.loss_fn(predictions, labels)

                    # Backward pass
                    adam_optimizer.zero_grad()
                    loss.backward()
                    adam_optimizer.step()


                elif self.use_info == "use hypothesis" or self.use_info == "known info noisy simulation":

                    overall_loss, individual_losses = loss_fn_custom(predictions, labels)
                    if DO.device == "cuda":
                        ind_loss_array = individual_losses.detach().cpu().numpy()
                    else:
                        ind_loss_array = individual_losses.detach().numpy()

                    if self.use_info != "known info noisy simulation":
                        partial_full_preds = model(DO.partial_input_tensor)
                        partial_incorr_full_preds = model(DO.inc_partial_input_tensor)     
                        DO.append2hyp_df(batch_i, ind_loss_array, "loss")                       

                        
                    if self.select_gradients == True:

                        if self.use_info == "known info noisy simulation":

                            grads = compute_individual_grads(model, individual_losses, DO.device)
                            DO.append2hyp_df(batch_i, grads, "gradients", layer = AM.layer)
                            DO.append2hyp_df(batch_i, ind_loss_array, "loss") 
                            # select gradients by avoiding noisy datapoints
                            sel_grads, sel_global_ids = gradient_selection_avoid_noise(AM, epoch, grads, self.batch_size, batch_i, inputs,
                                                                                       predictions, individual_losses) 
                            
                            if epoch >= self.num_epochs - self.end_epochs_noise_detection:
                                sel_grads = grads
                                if first_load == True:

                                    model = initialize_model(DO, self.dataloader, model.hidden_size, self.rand_state, dropout = AM.dropout)
                                    model.load_state_dict(torch.load(self.weights_save_path))
                                    custom_optimizer = CustomAdam(model.parameters(), lr= AM.lr) #, weight_decay=0.005
                                    first_load = False

                            #print(len(sel_grads))
                        else:

                            partial_overall_loss, partial_individual_losses = loss_fn_custom(partial_full_preds, DO.partial_full_outcomes)
                            self.all_partial_ind_losses[f"epoch_{epoch}"].append(partial_individual_losses)

                            inc_partial_overall_loss = []
                            inc_partial_overall_loss, inc_partial_individual_losses = loss_fn_custom(partial_incorr_full_preds, duplicate_elements(DO.partial_full_outcomes, DO.num_hyp_comb-1).view(-1,1))  

                            #calculate all hypothesis gradients
                            #To optimize speed calculate only the last 2 layers of gradients instead of everything
                            hypothesis_grads = compute_individual_grads(model, individual_losses, DO.device)
                            DO.latest_partial_grads = compute_individual_grads(model, partial_individual_losses, DO.device)
                            DO.latest_inc_partial_grads = compute_individual_grads(model, inc_partial_individual_losses, DO.device)

                            custom_optimizer.zero_grad()
                            overall_loss.backward()

                            DO.append2hyp_df(batch_i, hypothesis_grads, "gradients", layer = AM.layer)


                            #if AM.rmv_avg_grad_signal:

                            # select gradients
                            #print(AM.no_selection_epochs)
                            if epoch >= AM.no_selection_epochs:

                                #hypothesis_grads   inc_partial_full_grads

                                sel_grads, selected_global_ids = gradient_selection(DO, AM, epoch, hypothesis_grads, DO.latest_partial_grads, 
                                                                                self.batch_size, DO.num_hyp_comb, batch_i, 
                                                                                inputs, DO.partial_input_tensor, labels, predictions,
                                                                                DO.partial_full_outcomes, partial_full_preds, 
                                                                                individual_losses, partial_individual_losses, 
                                                                                DO.inc_partial_input_tensor, partial_incorr_full_preds,
                                                                                inc_partial_individual_losses, DO.latest_inc_partial_grads, self.rand_state)

                            else:
                                #Baseline Random Gradient selection from each group
                                sel_grads = gradient_random_selection(hypothesis_grads, DO.num_hyp_comb)

                        if self.use_info != "known info noisy simulation" and AM.partial_freq_per_epoch != 0 and batch_i % int(len(DO.df_train_hypothesis)/self.batch_size/AM.partial_freq_per_epoch) == 0:
                            sel_grads = sel_grads+DO.latest_partial_grads

                        # Update the weights
                        if len(sel_grads) == 0:
                            print("No gradients were selected, training will cease.")
                            break
                        overall_sel_grads = gradients_mean(sel_grads)
                        custom_optimizer.step(overall_sel_grads)
                        self.sel_grads_num_logs.append(len(sel_grads))
                    else:
                        adam_optimizer.zero_grad()
                        overall_loss.backward()
                        adam_optimizer.step()

            if epoch == self.num_epochs or ('sel_grads' in locals() and len(sel_grads) == 0):
                break  
            if self.use_info in ["full info", "use known only", "partial info", "use imputation", "full info noisy"]:    
                train_errors.append(loss.item())
            elif self.use_info in ["use hypothesis", "known info noisy simulation"]:
                train_errors.append(overall_loss.item())

            # Call the validation function
            if (epoch) % (self.num_epochs//num_validations) == 0 or epoch == (self.num_epochs-1): 
                if self.use_info in ["full info", "partial info", "use hypothesis", "use imputation", "known info noisy simulation", "full info noisy"]:
                    valid_error = self._validate_model(DO.full_val_input_tensor, DO.val_outcomes_tensor, model)
                elif self.use_info in ["use known only"]:
                    valid_error = self._validate_model(DO.known_val_input_tensor, DO.val_outcomes_tensor, model)
                #valid_error = np.mean(valid_error)
                if valid_error < self.best_valid_error:
                    self.best_valid_error = valid_error
                    torch.save(model.state_dict(), self.weights_save_path)
                    self.best_checkpoint = epoch
                    #if self.use_info in ["known info noisy simulation"]:
                    #    self.best_eps = eps
                    #    self.min_samples_ratio = min_samples_ratio


                self.valid_errors_epoch.append(np.mean(valid_error))
                #print(np.mean(valid_error))
                self.val_epochs.append(epoch)

            self.train_errors_epoch.append(np.mean(train_errors))
            
            
    def _validate_model(self, validation_inputs, validation_labels, model):

        model.eval()
        validation_predictions = model(validation_inputs)
        validation_loss = self.loss_fn(validation_predictions, validation_labels)
        model.train()

        return validation_loss.item()
    
    def _test_model(self, test_inputs, test_labels, model):

        model.eval()
        test_predictions = model(test_inputs)
        test_loss = self.loss_fn(test_predictions, test_labels)
        model.train()

        return test_loss.item()  
    
def range_with_tqdm(num_epochs, use_tqdm):
    if use_tqdm:
        return tqdm(range(num_epochs))
    else:
        return range(num_epochs)