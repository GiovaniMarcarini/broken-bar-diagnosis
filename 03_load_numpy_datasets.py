# Carrega os dados da classe para treino e teste:

stable = True # Set to True for latest pip version or False for main branch in GitHub
!pip install torch==2.5.1
#!pip install tsai -U
!pip install {"tsai -U" if stable else "git+https://github.com/timeseriesAI/tsai.git"} >> /dev/null
from tsai.all import *

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

temp_dir = '/content/drive/MyDrive/Mestrado/TempSplits/'
torque_levels = ["torque05", "torque10", "torque15", "torque20", "torque25", "torque30", "torque35", "torque40"]

# Carrega os dados da classe "rs" (motor saudável) para treino e teste:
X_train_rs, y_train_rs = load_class_splits(temp_dir, "struct_rs_R1", "rs", torque_levels, split_type="train")
X_test_rs,  y_test_rs  = load_class_splits(temp_dir, "struct_rs_R1", "rs", torque_levels, split_type="test")

# Carrega os dados da classe "r1b" (1 barra quebrada) para treino e teste:
X_train_r1b, y_train_r1b = load_class_splits(temp_dir, "struct_r1b_R1", "r1b", torque_levels, split_type="train")
X_test_r1b,  y_test_r1b  = load_class_splits(temp_dir, "struct_r1b_R1", "r1b", torque_levels, split_type="test")

# Carrega os dados da classe "r2b" (2 barra quebrada) para treino e teste:
X_train_r2b, y_train_r2b = load_class_splits(temp_dir, "struct_r2b_R1", "r2b", torque_levels, split_type="train")
X_test_r2b,  y_test_r2b  = load_class_splits(temp_dir, "struct_r2b_R1", "r2b", torque_levels, split_type="test")

# Carrega os dados da classe "r3b" (3 barra quebrada) para treino e teste:
X_train_r3b, y_train_r3b = load_class_splits(temp_dir, "struct_r3b_R1", "r3b", torque_levels, split_type="train")
X_test_r3b,  y_test_r3b  = load_class_splits(temp_dir, "struct_r3b_R1", "r3b", torque_levels, split_type="test")

# Carrega os dados da classe "r4b" (4 barra quebrada) para treino e teste:
X_train_r4b, y_train_r4b = load_class_splits(temp_dir, "struct_r4b_R1", "r4b", torque_levels, split_type="train")
X_test_r4b,  y_test_r4b  = load_class_splits(temp_dir, "struct_r4b_R1", "r4b", torque_levels, split_type="test")

# Combina os dados de treino de todas as classes:
X_train_all = np.concatenate([X_train_rs, X_train_r1b, X_train_r2b, X_train_r3b, X_train_r4b], axis=0)
y_train_all = np.concatenate([y_train_rs, y_train_r1b, y_train_r2b, y_train_r3b, y_train_r4b], axis=0)

# Combina os dados de teste de todas as classes:
X_test_all = np.concatenate([X_test_rs, X_test_r1b, X_test_r2b, X_test_r3b, X_test_r4b], axis=0)
y_test_all = np.concatenate([y_test_rs, y_test_r1b, y_test_r2b, y_test_r3b, y_test_r4b], axis=0)

from tsai.all import combine_split_data

# Combina os dados
X_all, y_all, splits = combine_split_data([X_train_all, X_test_all], [y_train_all, y_test_all])
