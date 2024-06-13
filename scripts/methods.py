import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import copy
#For reproducibility
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sklearn.model_selection import train_test_split

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  
        return x
    
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)
        self.fc3 = nn.Linear(10, 2)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  
        return x


def train_nn(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    val_data = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    patience, trials = 3, 0

    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        for i, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (X_batch, y_batch) in enumerate(val_loader):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                print('Early stopping!')
                break
    
    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        y_pred_val = model(torch.tensor(X_val).float())
        y_prob_val = nn.functional.softmax(y_pred_val, dim=1)
    return model, y_prob_val.numpy(), y_val



def irm_penalty(logits, y, scale):
    loss = nn.CrossEntropyLoss()(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def train_irm(X_train, y_train, g_train, penalty_scale):
    X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(X_train, y_train, g_train, test_size=0.1, random_state=42)
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    patience, trials = 3, 0

    train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long(), torch.tensor(g_train))
    val_data = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long(), torch.tensor(g_val))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    epochs = 50
    scale = torch.tensor(1.).requires_grad_()
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch, g_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = nn.CrossEntropyLoss()(logits, y_batch)

            penalty = irm_penalty(logits, y_batch, scale)
            loss += penalty_scale * penalty

            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                logits = model(X_batch)
                loss = nn.CrossEntropyLoss()(logits, y_batch)
                val_loss += loss.item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                print('Early stopping!')
                break

    model.load_state_dict(best_model_wts)
    return model

def train_group_dro(X_train, y_train, g_train):
    X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(X_train, y_train, g_train, test_size=0.1, random_state=42)
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    patience, trials = 3, 0

    train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long(), torch.tensor(g_train))
    val_data = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long(), torch.tensor(g_val))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch, g_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            losses = []

            unique_groups = torch.unique(g_batch)
            for g in unique_groups:
                group_mask = (g_batch == g)
                if group_mask.any():
                    loss = nn.CrossEntropyLoss()(logits[group_mask], y_batch[group_mask])
                    losses.append(loss)

            if losses:
                max_loss = max(losses)
                max_loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                logits = model(X_batch)
                loss = nn.CrossEntropyLoss()(logits, y_batch)
                val_loss += loss.item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                print('Early stopping!')
                break

    model.load_state_dict(best_model_wts)
    return model

def calculate_confusion_matrices(preds, targets, groups):
    probs = torch.softmax(preds, dim=1)
    
    unique_groups = torch.unique(groups)
    
    confusion_matrices = {g.item(): torch.zeros(2, 2, dtype=torch.float32) for g in unique_groups}
    
    for g in unique_groups:
        group_mask = (groups == g)
        for true_class in range(2):  
            true_mask = (targets == true_class) & group_mask
            for pred_class in range(2):
                confusion_matrices[g.item()][true_class, pred_class] = torch.sum(probs[true_mask, pred_class]) + 1e-6

    for g in unique_groups:
        row_sums = confusion_matrices[g.item()].sum(dim=1, keepdim=True)
        confusion_matrices[g.item()] = torch.div(confusion_matrices[g.item()], row_sums)

    return confusion_matrices


def train_ours(X_train, y_train, g_train, lambda_reg=1.0):
    X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(X_train, y_train, g_train, test_size=0.1, random_state=42)

    train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long(), torch.tensor(g_train))
    val_data = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long(), torch.tensor(g_val))
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=512, shuffle=False)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    patience, trials = 3, 0

    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch, g_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            reg_loss = torch.tensor(0.0)
            batch_conf_matrices = calculate_confusion_matrices(y_pred, y_batch, g_batch)
            groups = list(batch_conf_matrices.keys())
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    reg_loss += torch.norm(batch_conf_matrices[groups[i]] - batch_conf_matrices[groups[j]])
            reg_loss /= len(groups) * (len(groups) - 1) / 2
            loss = loss + lambda_reg * reg_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                print('Early stopping!')
                break

    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        y_pred_val = model(torch.tensor(X_val).float())
        y_prob_val = nn.functional.softmax(y_pred_val, dim=1)
    return model, y_prob_val.numpy(), y_val



def erm(X_train, y_train, X_test):
    model, y_prob_val, y_val = train_nn(X_train, y_train)
    X_test = torch.tensor(X_test).float()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_prob = nn.functional.softmax(y_pred, dim=1)
    return y_prob.numpy(), y_prob_val, y_val

def irm(X_train, y_train, g_train, X_test, penalty_scale):
    model = train_irm(X_train, y_train, g_train, penalty_scale)
    X_test = torch.tensor(X_test).float()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_prob = nn.functional.softmax(y_pred, dim=1)
    return y_prob.numpy()

def group_dro(X_train, y_train, g_train, X_test):
    model = train_group_dro(X_train, y_train, g_train)
    X_test = torch.tensor(X_test).float()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_prob = nn.functional.softmax(y_pred, dim=1)
    return y_prob.numpy()

def ours(X_train, y_train, g_train, X_test, lambda_reg=1.0):
    model, y_prob_val, y_val = train_ours(X_train, y_train, g_train, lambda_reg)
    X_test = torch.tensor(X_test).float()
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_prob = nn.functional.softmax(y_pred, dim=1)
    return y_prob.numpy(), y_prob_val, y_val, y_pred
