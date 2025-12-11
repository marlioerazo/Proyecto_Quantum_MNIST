import json
import numpy as np

# ==========================================
# 1. CARGAR DATOS
# ==========================================
print("Leyendo datos_finales.json...")
try:
    with open('datos_finales.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("ERROR: No se encontró 'datos_finales.json'. Ejecuta primero el script 01.")
    exit()

n_runs = data['config']['runs']
epochs = data['config']['epochs']
runs = data['runs']

# ==========================================
# 2. CÁLCULOS ESTADÍSTICOS RIGUROSOS
# ==========================================

# A. Curvas de Pérdida (Train y Val)
train_losses = np.array([r['train_loss'] for r in runs])
val_losses = np.array([r['val_loss'] for r in runs])

train_mean = np.mean(train_losses, axis=0)
train_std = np.std(train_losses, axis=0)
val_mean = np.mean(val_losses, axis=0)

# B. Métricas por Ejecución (Para tener Std Dev en TODAS)
acc_list = []
prec_list = []
rec_list = []
f1_list = []

for r in runs:
    # 1. Accuracy de esta corrida (última época)
    acc_list.append(r['val_acc'][-1])
    
    # 2. Métricas derivadas de la matriz de confusión de ESTA corrida
    cm = r['confusion_matrix']
    tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']
    
    # Cálculos protegidos contra división por cero
    p = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
    
    prec_list.append(p)
    rec_list.append(rec)
    f1_list.append(f)

# Convertir a arrays numpy
acc_arr = np.array(acc_list)
prec_arr = np.array(prec_list)
rec_arr = np.array(rec_list)
f1_arr = np.array(f1_list)

# C. Matriz de Confusión Promedio (Visualización)
# Promediamos los conteos para el dibujo de la matriz
tps = [r['confusion_matrix']['tp'] for r in runs]
tns = [r['confusion_matrix']['tn'] for r in runs]
fps = [r['confusion_matrix']['fp'] for r in runs]
fns = [r['confusion_matrix']['fn'] for r in runs]

tp_avg = int(round(np.mean(tps)))
tn_avg = int(round(np.mean(tns)))
fp_avg = int(round(np.mean(fps)))
fn_avg = int(round(np.mean(fns)))

# ==========================================
# 3. GENERADORES LATEX
# ==========================================

def get_loss_plot():
    c_train = "".join([f"({i+1}, {train_mean[i]:.4f}) " for i in range(epochs)])
    c_upper = "".join([f"({i+1}, {train_mean[i]+train_std[i]:.4f}) " for i in range(epochs)])
    c_lower = "".join([f"({i+1}, {max(0, train_mean[i]-train_std[i]):.4f}) " for i in range(epochs)])
    c_val   = "".join([f"({i+1}, {val_mean[i]:.4f}) " for i in range(epochs)])
    
    return f"""
% GRÁFICA DE PÉRDIDA
\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=0.95\\linewidth, height=6cm,
    xlabel={{Época}}, ylabel={{Pérdida (MSE)}},
    xmin=1, xmax={epochs},
    ymin=0, ymax={np.max(train_mean)*1.2:.2f},
    grid=major,
    legend pos=north east,
    legend style={{font=\\footnotesize}}
]
\\addplot[name path=upper, draw=none, forget plot] coordinates {{ {c_upper} }};
\\addplot[name path=lower, draw=none, forget plot] coordinates {{ {c_lower} }};
\\addplot[quantumblue!30, opacity=0.5] fill between[of=upper and lower];
\\addlegendentry{{Desv. Est. ($\\pm 1\\sigma$)}}
\\addplot[color=quantumblue, thick, mark=*, mark size=1.5pt] coordinates {{ {c_train} }};
\\addlegendentry{{Train (Media)}}
\\addplot[color=quantumpink, thick, dashed, mark=triangle, mark size=1.5pt] coordinates {{ {c_val} }};
\\addlegendentry{{Val (Media)}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Curvas de pérdida con bandas de desviación estándar ($\\pm 1\\sigma$) sobre {n_runs} ejecuciones.}}
\\label{{fig:loss_curves}}
\\end{{figure}}
"""

def get_confusion_matrix():
    return f"""
% MATRIZ DE CONFUSIÓN
\\begin{{figure}}[H]
\\centering
\\begin{{tikzpicture}}
\\matrix[matrix of nodes, 
    nodes={{minimum width=2.5cm, minimum height=1.5cm, anchor=center, font=\\large}},
    column 1/.style={{nodes={{fill=quantumblue!20}}}},
    column 2/.style={{nodes={{fill=quantumpink!20}}}},
    row 1/.style={{nodes={{font=\\bfseries\\small}}}},
    row sep=-\\pgflinewidth, column sep=-\\pgflinewidth] (m) {{
    & Pred. 0 & Pred. 1 \\\\
    Real 0 & {tn_avg} & {fp_avg} \\\\
    Real 1 & {fn_avg} & {tp_avg} \\\\
}};
\\draw[thick] (m-1-2.north west) -- (m-3-2.south west);
\\draw[thick] (m-2-1.north west) -- (m-2-3.north east);
\\node[left=0.1cm of m-2-1, rotate=90, font=\\bfseries] {{Real}};
\\node[above=0.1cm of m-1-2, font=\\bfseries, xshift=1.2cm] {{Predicho}};
\\end{{tikzpicture}}
\\caption{{Matriz de confusión (media $\\pm$ desv. est. sobre {n_runs} ejecuciones).}}
\\label{{fig:confusion}}
\\end{{figure}}
"""

def get_metrics_table():
    # Helper para formatear fila con Media, Std y CI
    def format_row(name, arr):
        mean = np.mean(arr)
        std = np.std(arr)
        # Intervalo de confianza 95% (Z=1.96), clipeado a 100%
        ci_lower = max(0, mean - 1.96 * std)
        ci_upper = min(100, mean + 1.96 * std)
        return f"{name} & {mean:.1f}\\% & {std:.2f}\\% & [{ci_lower:.1f}\\%, {ci_upper:.1f}\\%] \\\\"

    return f"""
% TABLA DE MÉTRICAS COMPLETA
\\begin{{table}}[H]
\\centering
\\caption{{Resultados en conjunto de prueba ({n_runs} ejecuciones)}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Métrica}} & \\textbf{{Media}} & \\textbf{{Desv. Est.}} & \\textbf{{IC 95\\%}} \\\\
\\midrule
{format_row("Accuracy", acc_arr)}
{format_row("Precision (clase 1)", prec_arr)}
{format_row("Recall (clase 1)", rec_arr)}
{format_row("F1-Score", f1_arr)}
\\bottomrule
\\end{{tabular}}
\\label{{tab:metrics}}
\\end{{table}}
"""

# ==========================================
# GUARDAR ARCHIVOS
# ==========================================
print("Generando archivos LaTeX corregidos...")
with open("latex_grafica.txt", "w", encoding="utf-8") as f: f.write(get_loss_plot())
with open("latex_matriz.txt", "w", encoding="utf-8") as f: f.write(get_confusion_matrix())
with open("latex_tabla.txt", "w", encoding="utf-8") as f: f.write(get_metrics_table())

print("\n¡HECHO! Abre 'latex_tabla.txt' y verás que todas las filas tienen datos completos.")
