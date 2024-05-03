#ml_py38 (3.8.16)
#torch.__version__ '2.0.1'

import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

from .custom_optimizer import CustomAdam

from .data_ops import flatten_list

import os
import glob
import json
import numpy as np
from sklearn.cluster import DBSCAN


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout = False, problem_type = "regression"):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size) #
        #self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        self.type = problem_type
        self.dropout = dropout
        if dropout:
            self.dropout1 = nn.Dropout(dropout)
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        
        if self.type  == "binary-class":
            x = self.sigmoid(x)
        elif self.type  == "multi-class":
            x = self.softmax(x)
        
        #x = self.relu(x)
        #x = self.fc4(x)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 12),
            nn.ReLU(True),
            nn.Linear(12, 8),
            nn.ReLU(True),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Linear(8, 12),
            nn.ReLU(True),
            nn.Linear(12, 4)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def loss(self, x):
        output = self(x)
        criterion = nn.MSELoss()
        x = x.view(-1, 4)  # reshape x to be [batch_size, 4]
        output = output.view(-1, 4)  # reshape output to be [batch_size, 4]
        loss_features = criterion(output[:, :3], x[:, :3])
        loss_outcome = criterion(output[:, 3], x[:, 3])
        return loss_features + 3 * loss_outcome  # adjust the weights as needed
    

def initialize_model(DO, dataloader, hidden_size, rand_state, dropout = False):

    tensor_batch = next(iter(dataloader))
    input_size = tensor_batch[0].shape[1]
    torch.manual_seed(rand_state)
    if not dropout:
        if len(DO.df_train) < 300:
            dropout = 0.10
      
    model = MLP(input_size, hidden_size, len(DO.target_vars), dropout, DO.problem_type)
    model = model.to(DO.device)
    return model
    
def validate_model(validation_inputs, validation_labels, model, loss_fn):

    model.eval()
    validation_predictions = model(validation_inputs)
    validation_loss, _ = loss_fn(validation_predictions, validation_labels)
    model.train()

    return validation_loss.item()

def load_model(DO, model_path, batch_size):
    
    #print("".join(model_path.split("/")[:-1]))  
    json_files = glob.glob(os.path.join("/".join(model_path.split("/")[:-1]), "*.json"))
    #print(json_files)
    if json_files:
        with open(json_files[-1]) as f:
            json_file = json.load(f)

        dataloader = DO.prep_dataloader(json_file["info_used"], batch_size)    
        model = initialize_model(DO, dataloader, json_file["hidden_size"], DO.rand_state, dropout = json_file["model_dropout"])
        model.load_state_dict(torch.load(model_path))
    
        return model
    
    else:
        raise ValidationError("No json file with details found, when trying to load model.")