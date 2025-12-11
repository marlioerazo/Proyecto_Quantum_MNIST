import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pennylane as qml
import numpy as np

# ==========================================
# CONFIGURACIÓN
# ==========================================
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.0001 # Un poco más rápido para ver cambios claros en 10 épocas
N_QUBITS = 3
SEED = 42
device = torch.device("cpu")

# ==========================================
# 1. DATOS
# ==========================================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
idx = (train_dataset.targets == 0) | (train_dataset.targets == 1)
train_dataset.targets = train_dataset.targets[idx]
train_dataset.data = train_dataset.data[idx]
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Imagen fija para monitorear el entrelazamiento
fixed_input = train_dataset[0][0].unsqueeze(0).to(device)

# ==========================================
# 2. CIRCUITOS (Uno para entrenar, otro para medir)
# ==========================================
dev_train = qml.device("default.qubit", wires=N_QUBITS)
dev_analysis = qml.device("default.qubit", wires=N_QUBITS)

def circuit_core(inputs, weights):
    # Capa Hadamard
    for q in range(N_QUBITS): qml.Hadamard(wires=q)
    # Feature Map
    for q in range(N_QUBITS): qml.RY(inputs[q], wires=q)
    # Entrelazamiento
    qml.CNOT(wires=[0, 1]); qml.CNOT(wires=[1, 2]); qml.CNOT(wires=[2, 0])
    # Variacional
    for q in range(N_QUBITS): qml.RZ(weights[q], wires=q)

# QNode A: Para entrenar (Devuelve valor esperado)
@qml.qnode(dev_train, interface="torch", diff_method="backprop")
def training_circuit(inputs, weights):
    circuit_core(inputs, weights)
    return qml.expval(qml.PauliZ(0))

# QNode B: Para analizar (Devuelve matriz densidad)
# Importante: interface='torch' para que sea compatible con los tensores
@qml.qnode(dev_analysis, interface="torch")
def get_density_matrix_0(inputs, weights):
    circuit_core(inputs, weights)
    return qml.density_matrix(wires=0)

@qml.qnode(dev_analysis, interface="torch")
def get_density_matrix_1(inputs, weights):
    circuit_core(inputs, weights)
    return qml.density_matrix(wires=1)

# ==========================================
# 3. MODELO REAL
# ==========================================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 3) 
        # Inicializamos pesos pequeños para empezar con poco entrelazamiento
        self.q_params = nn.Parameter(0.05 * torch.randn(3))
        
    def get_angles(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.pi * torch.tanh(x)

    def forward(self, x):
        angles = self.get_angles(x)
        # Ejecutamos circuito de entrenamiento
        exp_vals = [training_circuit(a, self.q_params) for a in angles]
        exp_vals = torch.stack(exp_vals).unsqueeze(1)
        return torch.sigmoid(exp_vals.float() * 5)

# Función auxiliar para calcular entropía desde matriz densidad
def calculate_entropy(rho):
    if isinstance(rho, torch.Tensor):
        rho = rho.detach().cpu().numpy()
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals, 1e-10, 1.0)
    return -np.sum(eigvals * np.log2(eigvals))

# ==========================================
# 4. EJECUCIÓN DEL ANÁLISIS
# ==========================================
def run_entropy_analysis():
    print("Iniciando entrenamiento REAL para medir entrelazamiento...")
    torch.manual_seed(SEED)
    model = HybridModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    hist_s0 = []
    hist_s1 = []
    
    # Medición inicial (Antes de entrenar)
    with torch.no_grad():
        angles = model.get_angles(fixed_input)[0]
        hist_s0.append(calculate_entropy(get_density_matrix_0(angles, model.q_params)))
        hist_s1.append(calculate_entropy(get_density_matrix_1(angles, model.q_params)))

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Bucle de entrenamiento REAL
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Medir entropía al final de la época
        with torch.no_grad():
            angles = model.get_angles(fixed_input)[0]
            s0 = calculate_entropy(get_density_matrix_0(angles, model.q_params))
            s1 = calculate_entropy(get_density_matrix_1(angles, model.q_params))
            hist_s0.append(s0)
            hist_s1.append(s1)
            
            avg_loss = total_loss / len(train_loader)
            print(f"Ep {epoch+1}: Loss={avg_loss:.4f} | S(rho0)={s0:.3f}, S(rho1)={s1:.3f}")

    # ==========================================
    # GENERAR LATEX
    # ==========================================
    coords_0 = "".join([f"({i}, {val:.4f}) " for i, val in enumerate(hist_s0)])
    coords_1 = "".join([f"({i}, {val:.4f}) " for i, val in enumerate(hist_s1)])
    
    latex_code = f"""
% FIGURA DE ENTRELAZAMIENTO
\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=0.95\\linewidth, height=5cm,
    xlabel={{Época}}, ylabel={{Entropía de von Neumann (bits)}},
    xmin=0, xmax={EPOCHS},
    ymin=0, ymax={max(np.max(hist_s0), np.max(hist_s1))*1.2:.2f},
    grid=major,
    legend pos=south east,
    legend style={{font=\\footnotesize}}
]
% Curva Qubit 0 (Morada)
\\addplot[color=quantumpurple, thick, mark=*,mark size=1.5pt] coordinates {{
    {coords_0}
}};
\\addlegendentry{{$S(\\rho_0)$}}

% Curva Qubit 1 (Verde)
\\addplot[color=quantumgreen, thick, dashed, mark=triangle,mark size=2pt] coordinates {{
    {coords_1}
}};
\\addlegendentry{{$S(\\rho_1)$}}

\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Evolución del entrelazamiento durante el entrenamiento. La dinámica muestra cómo el circuito ajusta las correlaciones cuánticas para minimizar la función de costo.}}
\\label{{fig:entanglement}}
\\end{{figure}}
"""
    
    with open("latex_grafica_entrelazamiento.txt", "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(" -> Generado: latex_grafica_entrelazamiento.txt")

if __name__ == "__main__":
    run_entropy_analysis()
