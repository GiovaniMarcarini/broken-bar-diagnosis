import numpy as np
import h5py
import os
import gc
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')
stable = True  # usar versão estável do tsai
!pip install torch==2.5.1 -q
!pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null

from tsai.all import *
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Função para carregar splits salvos (.npz)

def load_class_splits(temp_dir, base_name, group, torque_list, split_type="train"):
    """
    Carrega e concatena os splits salvos para uma classe.
    split_type: "train" ou "test"
    """
    X_list = []
    y_list = []
    print(f"Carregando splits para a base '{base_name}' do grupo '{group}' com split '{split_type}'...")
    
    for torque in torque_list:
        file_path = os.path.join(temp_dir, f"{base_name}_{group}_{torque}_{split_type}.npz")
        print(f"  Verificando arquivo para {torque}: {file_path}")
        
        if os.path.exists(file_path):
            data = np.load(file_path)
            if split_type == "train":
                X_list.append(data["X_train"])
                y_list.append(data["y_train"])
                print(f"    Carregado: X_train shape = {data['X_train'].shape}, y_train shape = {data['y_train'].shape}")
            else:
                X_list.append(data["X_test"])
                y_list.append(data["y_test"])
                print(f"    Carregado: X_test shape = {data['X_test'].shape}, y_test shape = {data['y_test'].shape}")
        else:
            print(f"    Arquivo não encontrado: {file_path}")
    
    if X_list:
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        print(f"Splits concatenados para '{base_name}' ({group}), {split_type}: X shape = {X.shape}, y shape = {y.shape}")
        return X, y
    else:
        print("Nenhum split carregado. Retornando None.")
        return None, None

# 2. Carrega todos os splits LIMPOS

temp_dir = '/content/drive/MyDrive/Mestrado/3- Segmentos_para_treinamento/'
torque_levels = ["torque05","torque20", "torque25","torque35", "torque40"]

# "rs" (saudável)
X_train_rs, y_train_rs = load_class_splits(temp_dir, "struct_rs_R1",  "rs",  torque_levels, split_type="train")
X_test_rs,  y_test_rs  = load_class_splits(temp_dir, "struct_rs_R1",  "rs",  torque_levels, split_type="test")

# "r1b"
X_train_r1b, y_train_r1b = load_class_splits(temp_dir, "struct_r1b_R1", "r1b", torque_levels, split_type="train")
X_test_r1b,  y_test_r1b  = load_class_splits(temp_dir, "struct_r1b_R1", "r1b", torque_levels, split_type="test")

# "r2b"
X_train_r2b, y_train_r2b = load_class_splits(temp_dir, "struct_r2b_R1", "r2b", torque_levels, split_type="train")
X_test_r2b,  y_test_r2b  = load_class_splits(temp_dir, "struct_r2b_R1", "r2b", torque_levels, split_type="test")

# "r3b"
X_train_r3b, y_train_r3b = load_class_splits(temp_dir, "struct_r3b_R1", "r3b", torque_levels, split_type="train")
X_test_r3b,  y_test_r3b  = load_class_splits(temp_dir, "struct_r3b_R1", "r3b", torque_levels, split_type="test")

# "r4b"
X_train_r4b, y_train_r4b = load_class_splits(temp_dir, "struct_r4b_R1", "r4b", torque_levels, split_type="train")
X_test_r4b,  y_test_r4b  = load_class_splits(temp_dir, "struct_r4b_R1", "r4b", torque_levels, split_type="test")

# 3. Combina todas as classes (limpo)

X_train_all = np.concatenate([X_train_rs, X_train_r1b, X_train_r2b, X_train_r3b, X_train_r4b], axis=0)
y_train_all = np.concatenate([y_train_rs, y_train_r1b, y_train_r2b, y_train_r3b, y_train_r4b], axis=0)

X_test_all = np.concatenate([X_test_rs, X_test_r1b, X_test_r2b, X_test_r3b, X_test_r4b], axis=0)
y_test_all = np.concatenate([y_test_rs, y_test_r1b, y_test_r2b, y_test_r3b, y_test_r4b], axis=0)

print("\nShapes LIMPOS:")
print("  X_train_all:", X_train_all.shape)
print("  y_train_all:", y_train_all.shape)
print("  X_test_all: ", X_test_all.shape)
print("  y_test_all: ", y_test_all.shape)

# (Opcional) Split base LIMPA
X_all_clean, y_all_clean, splits_clean = combine_split_data(
    [X_train_all, X_test_all],
    [y_train_all, y_test_all]
)
print("\nBase combinada LIMPA:")
print("  X_all_clean:", X_all_clean.shape)
print("  y_all_clean:", y_all_clean.shape)
print("  splits_clean:", [len(s) for s in splits_clean])  # [n_train, n_test]

# 4. Funções de perturbação: AWGN e Missing

def awgn_nd(x, SNRdB):
    """
    Aplica ruído AWGN em um array N-dimensional (ex.: (N, C, L)).
    SNRdB: SNR desejado em dB.
    """
    P = np.mean(x ** 2)  # potência média
    if P == 0:
        return x.copy()

    gamma = 10 ** (SNRdB / 10.0)  # SNR linear
    N0 = P / gamma                # densidade espectral de ruído

    noise = np.sqrt(N0 / 2.0) * np.random.randn(*x.shape)
    return x + noise

def random_zeros_nd(x, p):
    """
    Insere zeros aleatórios em um array N-dimensional,
    cada posição tem probabilidade p de virar zero.
    """
    mask = np.random.choice([0, 1], size=x.shape, p=[p, 1-p])
    return x * mask

snr_list = [30, 20, 10, 0]# níveis de ruído em dB
missing_p_list = [0.1, 0.3, 0.5]  # proporções de missing (10%, 30%, 50%)

# 5. Define modelos a serem avaliados

# Caminhos dos modelos treinados
model_paths = {
    "TST":           "/content/drive/MyDrive/Mestrado/4 - Modelos_treinados/ArchTST/TST.pkl",
    "InceptionTime": "/content/drive/MyDrive/Mestrado/4 - Modelos_treinados/InceptionTime/InceptionTime.pkl",
    "LSTM":         "/content/drive/MyDrive/Mestrado/4 - Modelos_treinados/ArchLSTM/LSTM.pkl",
}

# Salvar as figuras geradas
save_dir = "/content/drive/MyDrive/Mestrado/4 - Modelos_treinados"
os.makedirs(save_dir, exist_ok=True)

# 6. Função de avaliação com métricas + matriz de confusão

def avaliar_modelo_em_conjunto(learn, X, y, titulo_fig, save_path):
    y_true = y.astype(int)

    probas, _, preds = learn.get_X_preds(X)
    preds_int = preds.astype(int)

    acc = skm.accuracy_score(y_true, preds_int)
    f1 = skm.f1_score(y_true, preds_int, average='weighted')
    precision = skm.precision_score(y_true, preds_int, average='weighted')
    recall = skm.recall_score(y_true, preds_int, average='weighted')
    balanced_acc = skm.balanced_accuracy_score(y_true, preds_int)
    kappa = skm.cohen_kappa_score(y_true, preds_int)

    print(f"\n================ {titulo_fig} ================")
    print("Acurácia:", acc)
    print("F1-Score (weighted):", f1)
    print("Precisão (weighted):", precision)
    print("Revocação (weighted):", recall)
    print("Balanced Accuracy:", balanced_acc)
    print("Cohen’s Kappa:", kappa)
    print("\nRelatório de Classificação:\n")
    print(skm.classification_report(y_true, preds_int))

    # Matriz de confusão normalizada
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display = skm.ConfusionMatrixDisplay.from_predictions(
        y_true, preds_int,
        normalize='true',
        cmap='Blues',
        ax=ax,
        colorbar=False
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    ax.set_xlabel("Predicted Class", fontsize=16)
    ax.set_ylabel("True Class", fontsize=16)
    ax.set_title(titulo_fig, fontsize=18)

    for text in cm_display.ax_.texts:
        text.set_fontsize(18)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "balanced_acc": balanced_acc,
        "kappa": kappa,
    }

# 7. Gerar todos os conjuntos de teste (clean, noise, missing)

test_sets = {}

# Base clean
test_sets["clean"] = (X_test_all, y_test_all)

# Bases com ruído
for snr in snr_list:
    X_noise = awgn_nd(X_test_all, snr)
    test_sets[f"noise_SNR{snr}dB"] = (X_noise, y_test_all)

# Bases com missing
for p in missing_p_list:
    pct = int(p * 100)
    X_missing = random_zeros_nd(X_test_all, p)
    test_sets[f"missing_{pct}pct"] = (X_missing, y_test_all)

print("\nConjuntos de teste gerados:")
for name, (X_ts, y_ts) in test_sets.items():
    print(f"  {name}: X={X_ts.shape}, y={y_ts.shape}")

# 8. Avaliar TODOS os modelos em todos os conjuntos

results = []  # para guardar métrica resumida por modelo/conjunto

for model_name, model_path in model_paths.items():
    print("\n" + "="*80)
    print(f"Avaliando modelo: {model_name}")
    print("="*80)

    learn = load_learner(model_path, cpu=True)
    learn.model.eval()

    for set_name, (X_ts, y_ts) in test_sets.items():
        titulo = f"{model_name} - {set_name}"
        fig_name = f"cm_{model_name}_{set_name}.png"
        save_path = os.path.join(save_dir, fig_name)

        metrics = avaliar_modelo_em_conjunto(
            learn, X_ts, y_ts,
            titulo_fig=titulo,
            save_path=save_path
        )

        row = {
            "model": model_name,
            "set": set_name,
            **metrics
        }
        results.append(row)

# 9. Tabela final com as métricas de todos
import pandas as pd

df_results = pd.DataFrame(results)
print("\n\n📊 RESUMO GERAL (todos os modelos x perturbações):")
print(df_results)
