import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pennylane as qml
import numpy as np
import json
import os

NUM_RUNS = 3
EPOCHS = 6
BATCH_SIZE = 64
LEARNING_RATE = 0.000001
N_QUBITS = 3
SEED = 42
device = torch.device("cpu")

def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    idx_train = (train_dataset.targets == 0) | (train_dataset.targets == 1)
    idx_test = (test_dataset.targets == 0) | (test_dataset.targets == 1)
    train_dataset.targets = train_dataset.targets[idx_train]
    train_dataset.data = train_dataset.data[idx_train]
    test_dataset.targets = test_dataset.targets[idx_test]
    test_dataset.data = test_dataset.data[idx_test]
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    for q in range(N_QUBITS): qml.Hadamard(wires=q)
    for q in range(N_QUBITS): qml.RY(inputs[q], wires=q)
    qml.CNOT(wires=[0, 1]); qml.CNOT(wires=[1, 2]); qml.CNOT(wires=[2, 0])
    for q in range(N_QUBITS): qml.RZ(weights[q], wires=q)
    return qml.expval(qml.PauliZ(0))

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 3) 
        self.q_params = nn.Parameter(0.01 * torch.randn(3))
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        opt_angles = torch.pi * torch.tanh(x)
        exp_vals = [quantum_circuit(a, self.q_params) for a in opt_angles]
        exp_vals = torch.stack(exp_vals).unsqueeze(1)
        return torch.sigmoid(exp_vals.float() * 5)

def run_full_experiment():
    print(f"Iniciando experimento: {NUM_RUNS} runs, {EPOCHS} epochs.")
    experiment_data = {"config": {"epochs": EPOCHS, "runs": NUM_RUNS}, "runs": []}
    for run in range(NUM_RUNS):
        print(f"Run {run + 1}/{NUM_RUNS}")
        torch.manual_seed(SEED + run)
        model = HybridModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()
        train_loader, test_loader = get_data_loaders()
        run_history = {"train_loss": [], "val_loss": [], "val_acc": [], "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}}
        for epoch in range(EPOCHS):
            model.train()
            tl = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                tl += loss.item()
            model.eval()
            vl = 0; corr = 0; tot = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device).float().unsqueeze(1)
                    output = model(data)
                    vl += criterion(output, target).item()
                    pred = (output > 0.5).float()
                    corr += (pred == target).sum().item()
                    tot += target.size(0)
            run_history["train_loss"].append(tl / len(train_loader))
            run_history["val_loss"].append(vl / len(test_loader))
            run_history["val_acc"].append(100 * corr / tot)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device).float().unsqueeze(1)
                output = model(data)
                pred = (output > 0.5).float().view(-1); y = target.view(-1)
                for p, t in zip(pred, y):
                    if p==1 and t==1: run_history["confusion_matrix"]["tp"] += 1
                    elif p==0 and t==0: run_history["confusion_matrix"]["tn"] += 1
                    elif p==1 and t==0: run_history["confusion_matrix"]["fp"] += 1
                    elif p==0 and t==1: run_history["confusion_matrix"]["fn"] += 1
        experiment_data["runs"].append(run_history)
    with open('datos_finales.json', 'w') as f: json.dump(experiment_data, f)

if __name__ == "__main__":
    run_full_experiment()
