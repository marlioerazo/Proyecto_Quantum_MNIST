import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pennylane as qml
import numpy as np
import json
import os

# ==========================================
# CONFIGURACIÓN (Optimizada para curva didactica))
# ==========================================
NUM_RUNS = 3            # Estadística suficiente
EPOCHS = 6              # Duración perfecta para ver la curva
BATCH_SIZE = 64         # Rápido
LEARNING_RATE = 0.000001 # Velocidad de aprendizaje suave
N_QUBITS = 3
SEED = 42
device = torch.device("cpu")

# ==========================================
# PREPARACIÓN DE DATOS
# ==========================================
def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Filtrar 0 y 1
    idx_train = (train_dataset.targets == 0) | (train_dataset.targets == 1)
    idx_test = (test_dataset.targets == 0) | (test_dataset.targets == 1)
    
    train_dataset.targets = train_dataset.targets[idx_train]
    train_dataset.data = train_dataset.data[idx_train]
    test_dataset.targets = test_dataset.targets[idx_test]
    test_dataset.data = test_dataset.data[idx_test]
    
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# CIRCUITO Y MODELO
# ==========================================
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
        # .float() CORRIGE EL ERROR DE TIPOS
        return torch.sigmoid(exp_vals.float() * 5)

# ==========================================
# BUCLE DE ENTRENAMIENTO
# ==========================================
def run_full_experiment():
    print(f"Iniciando experimento robusto: {NUM_RUNS} ejecuciones de {EPOCHS} épocas.")
    
    # Estructura para guardar TODO
    experiment_data = {
        "config": {"epochs": EPOCHS, "runs": NUM_RUNS},
        "runs": [] # Lista de diccionarios por corrida
    }
    
    for run in range(NUM_RUNS):
        print(f"\n>>> Ejecución {run + 1}/{NUM_RUNS}")
        torch.manual_seed(SEED + run)
        
        model = HybridModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()
        train_loader, test_loader = get_data_loaders()
        
        # Historial de ESTA corrida
        run_history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        }
        
        for epoch in range(EPOCHS):
            # --- TRAIN ---
            model.train()
            train_loss_acc = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device).float().unsqueeze(1)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss_acc += loss.item()
            avg_train_loss = train_loss_acc / len(train_loader)
            
            # --- VALIDATION ---
            model.eval()
            val_loss_acc = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device).float().unsqueeze(1)
                    output = model(data)
                    # Loss
                    loss = criterion(output, target)
                    val_loss_acc += loss.item()
                    # Acc
                    predicted = (output > 0.5).float()
                    correct += (predicted == target).sum().item()
                    total += target.size(0)
            
            avg_val_loss = val_loss_acc / len(test_loader)
            current_acc = 100 * correct / total
            
            # Guardar datos de la época
            run_history["train_loss"].append(avg_train_loss)
            run_history["val_loss"].append(avg_val_loss)
            run_history["val_acc"].append(current_acc)
            
            print(f"Ep {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Acc={current_acc:.2f}%")
        
        # --- MATRIZ DE CONFUSIÓN FINAL ---
        # Calculamos la matriz al final de la corrida
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device).float().unsqueeze(1)
                output = model(data)
                pred = (output > 0.5).float().view(-1)
                y = target.view(-1)
                for p, t in zip(pred, y):
                    if p==1 and t==1: run_history["confusion_matrix"]["tp"] += 1
                    elif p==0 and t==0: run_history["confusion_matrix"]["tn"] += 1
                    elif p==1 and t==0: run_history["confusion_matrix"]["fp"] += 1
                    elif p==0 and t==1: run_history["confusion_matrix"]["fn"] += 1
        
        experiment_data["runs"].append(run_history)

    # Guardar en JSON
    with open('datos_finales.json', 'w') as f:
        json.dump(experiment_data, f)
    print("\n[OK] Datos guardados en 'datos_finales.json'. Ejecuta el siguiente script.")

if __name__ == "__main__":
    run_full_experiment()
