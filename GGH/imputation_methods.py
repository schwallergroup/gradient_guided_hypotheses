import pandas as pd
import numpy as np
import torch

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from fancyimpute import NuclearNormMinimization, SoftImpute, BiScaler, MatrixFactorization

import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
import MIDASpy as md
from hyperimpute.plugins.imputers import Imputers

from .models import initialize_model
from .train_val_loop import TrainValidationManager

import warnings

        

class Imputer():
    def __init__(self, DO):
        self.best_imputer = "undetermined"
        self.rand_state = DO.rand_state 
        self.df_train_imp = prep_imputation_input(DO, DO.df_train[DO.inpt_vars+DO.miss_vars+DO.target_vars], DO.miss_vars, DO.partial_rows_id)
        self.imput_input = self.df_train_imp.values
        
        self.imputers = {
                            "Iterative MICE Imputer": IterativeImputer(), #max_iter=30, initial_strategy="most_frequent"
                            "KNN Imputer": KNNImputer(n_neighbors=3),
                            "Miss Forest": MissForest(),
                            "Deep Regressor": DeepRegressor(num_epochs = 100, lr= 0.001, rand_state = self.rand_state),
                            "Soft Impute": SoftImpute(),
                            "Matrix Factorization": MatrixFactorization(max_iters=100),
                            "Hyperimpute": HyperimputeWrapper()
                            #"Midas": "Midas" #midas was used to generate results in manuscript
                            
                            #("Nuclear Norm. Minimization", NuclearNormMinimization()),
                            #("BiScaler", BiScaler()),                     
        }
        self.results_dict = {}
        
    def test_all_imputers(self, DO, AM, INSPECT, batch_size, hidden_size, num_epochs, results_path): 
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
        for name, imputer in self.imputers.items():
            print(f"Testing {name}")
            imputer_failed = False
            if name != "Midas":           
                try:
                    imput_input = imputer.fit_transform(self.imput_input)
                    print("imputed")
                except:
                    imputer_failed = True
                    print("imput failed")
            else:
                df_train_imp, imputations = imput_w_midas(DO, self.df_train_imp, self.rand_state)
                #print(type(imputations))
                #print(imputations)
                #imput_input = imputations[0].values
                imput_input = df_train_imp.values
            
            if imputer_failed:
                self.results_dict[name] = "Method couldn't compute."
            else:
                print(type(imput_input))
                print(imput_input)
                imput_input = imput_input[:,:-1]

                dataloader = DO.prep_dataloader("use imputation", batch_size, imputed_input = imput_input)

                model = initialize_model(DO, dataloader, hidden_size, self.rand_state, dropout = 0.1) 
                TVM = TrainValidationManager("use imputation", num_epochs, dataloader, batch_size, self.rand_state, results_path)
                TVM.train_model(DO, AM, model)
                self.results_dict[name] = np.min(TVM.valid_errors_epoch) #INSPECT.calculate_val_r2score(DO, TVM, model)
                INSPECT.save_train_val_logs(DO, AM, TVM, model, imput_method = name)
        self.best_imputer = min(self.results_dict, key=self.results_dict.get)
                
    def impute_w_best(self, **kwargs):
        if self.best_imputer == "undetermined":
            raise ValueError("Best imputer has not been determined yet.")
        if self.best_imputer != "Midas":    
            imput_input = imputer.fit_transform(self.imput_input)
        else:
            df_train_imp, imputations = imput_w_midas(DO, self.df_train_imp, self.rand_state)
            imput_input = df_train_imp.values
       
        imput_input = imput_input[:,:-1]
        dataloader = DO.prep_dataloader("use imputation", batch_size, imputed_input = imput_input)
        return dataloader
    
    def impute_w_sel(self, DO, method, batch_size):
        if method != "Midas":    
            imput_input = self.imputers[method].fit_transform(self.imput_input)
        else:
            df_train_imp, imputations = imput_w_midas(DO, self.df_train_imp, self.rand_state)
            imput_input = df_train_imp.values
       
        imput_input = imput_input[:,:-1]
        dataloader = DO.prep_dataloader("use imputation", batch_size, imputed_input = imput_input)
        return dataloader

        
def prep_imputation_input(DO, df: pd.DataFrame, col_missing_data: str, row_indexes: list) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy.loc[~df_copy.index.isin(row_indexes), col_missing_data] = float('NaN')
    return df_copy[DO.inpt_vars+DO.miss_vars+DO.target_vars]   

class DeepRegressor():
    def __init__(self, num_epochs, lr, rand_state):
        self.num_epochs = num_epochs
        self.lr = lr
        self.rand_state = rand_state
    def impute_matrix(self, matrix):
        torch.manual_seed(self.rand_state)
        
        # define the mask of missing values
        mask = np.isnan(matrix)
    
        # find the column with missing values
        column = np.argwhere(np.sum(mask, axis=0) > 0)[0]
    
        # separate complete and incomplete rows
        complete_rows = matrix[~mask[:, column].reshape(mask[:, column].shape[0]), :]
        incomplete_rows = matrix[mask[:, column].reshape(mask[:, column].shape[0]), :]
    
        # convert data to tensors
        complete_rows_tensor_input = torch.from_numpy(np.delete(complete_rows, column, axis=1)).float()
        complete_rows_tensor_output = torch.from_numpy(complete_rows[:,column]).float()
        
        incomplete_rows_tensor_input = torch.from_numpy(np.delete(incomplete_rows, column, axis=1)).float()
    
        # define the neural network architecture
        input_dim = matrix.shape[1]-1
        hidden_dim = input_dim
        output_dim = 1
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
    
        # train the model
        for epoch in range(self.num_epochs):
            # forward pass
            outputs = model(complete_rows_tensor_input)
            # compute the loss
            loss = criterion(outputs, complete_rows_tensor_output)
            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # use the trained model to impute missing values
        imputed_rows_tensor = model(incomplete_rows_tensor_input)
        imputed_rows = imputed_rows_tensor.detach().numpy()
    
        # combine complete and imputed rows
        matrix_imputed = matrix.copy()
        matrix_imputed[mask[:, column].reshape(mask[:, column].shape[0]), column] = imputed_rows.reshape(imputed_rows.shape[0])
        #matrix_imputed[mask[:, column].reshape(mask[:, column].shape[0]), :] = imputed_rows
    
        return matrix_imputed

    def fit_transform(self, matrix):
        return self.impute_matrix(matrix)
    
def imput_w_midas(DO, df_train_imp, rand_state):

    na_loc = df_train_imp.isnull()
    df_train_imp[na_loc] = np.nan
    df_train_imp.reset_index(inplace = True, drop = True)
    imputer = md.Midas(layer_structure = [32,32], vae_layer = False, seed = rand_state, input_drop = 0.70)
    imputer.build_model(df_train_imp) #, softmax_columns = cat_cols_list
    imputer.train_model(training_epochs = 50)
    imputations = imputer.generate_samples(m=12).output_list
    model = md.combine(y_var = DO.miss_vars[0], 
                       X_vars = DO.inpt_vars+DO.target_vars,
                       df_list = imputations)

    df_train_imp_X = df_train_imp[DO.inpt_vars+DO.target_vars][df_train_imp[DO.miss_vars[0]].isnull()]

    imputed_value = model['estimate'][0]  # Start with the intercept
    for i, col_name in enumerate(DO.inpt_vars+DO.target_vars):
        imputed_value += df_train_imp_X[col_name] * model['estimate'][i + 1]

    df_train_imp.loc[df_train_imp[DO.miss_vars[0]].isnull(), DO.miss_vars[0]] = imputed_value

    return df_train_imp, imputations

class HyperimputeWrapper():

    def __init__(self):
        
        self.plugin = Imputers().get(
                        "hyperimpute",
                        optimizer="hyperband",
                        classifier_seed=["logistic_regression", "catboost", "xgboost", "random_forest"],
                        regression_seed=[
                            "linear_regression",
                            "catboost_regressor",
                            "xgboost_regressor",
                            "random_forest_regressor",
                        ],
                        # class_threshold: int. how many max unique items must be in the column to be is associated with categorical
                        class_threshold=5,
                        # imputation_order: int. 0 - ascending, 1 - descending, 2 - random
                        imputation_order=2,
                        # n_inner_iter: int. number of imputation iterations
                        n_inner_iter=10,
                        # select_model_by_column: bool. If true, select a different model for each column. Else, it reuses the model chosen for the first column.
                        select_model_by_column=True,
                        # select_model_by_iteration: bool. If true, selects new models for each iteration. Else, it reuses the models chosen in the first iteration.
                        select_model_by_iteration=True,
                        # select_lazy: bool. If false, starts the optimizer on every column unless other restrictions apply. Else, if for the current iteration there is a trend(at least to columns of the same type got the same model from the optimizer), it reuses the same model class for all the columns without starting the optimizer.
                        select_lazy=True,
                        # select_patience: int. How many iterations without objective function improvement to wait.
                        select_patience=5,
                    )

        
    def fit_transform(self, X):
        imputed = self.plugin.fit_transform(X.copy())
        return imputed.values