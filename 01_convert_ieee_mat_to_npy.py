import numpy as np
import h5py
import os
import gc
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')

# Diretório onde estão os arquivos .mat
base_dir = '/content/drive/MyDrive/Mestrado/Bases/'

# Diretório onde serão salvos os arquivos .npy convertidos
output_dir = '/content/drive/MyDrive/Mestrado/Converted/'
os.makedirs(output_dir, exist_ok=True)

# Lista dos níveis de torque a serem processados
torque_levels = ["torque05", "torque10", "torque15", "torque20", "torque25", "torque30", "torque35", "torque40"]

# Lista de bases com os nomes dos arquivos e o grupo correspondente
bases = [
    ("struct_rs_R1.mat", "rs"),
    ("struct_r1b_R1.mat", "r1b"),
    ("struct_r2b_R1.mat", "r2b"),
    ("struct_r3b_R1.mat", "r3b"),
    ("struct_r4b_R1.mat", "r4b")
]

def load_mat_raw(file_path, group, torque_key, signal_key="Ia"):
    """
    Carrega os dados de um arquivo MAT (formato HDF5) para um determinado grupo, nível de torque e sinal.

    Retorna:
      Array NumPy com os dados do sinal.
    """
    with h5py.File(file_path, 'r') as f:
        dset = f[group][torque_key][signal_key]
        data_obj = np.array(dset[:])

        if data_obj.dtype == np.object_:
            resolved_list = []
            for ref in data_obj.ravel():
                resolved_data = np.array(f[ref][:])
                if resolved_data.ndim == 2 and resolved_data.shape[0] == 1:
                    resolved_data = resolved_data.flatten()
                resolved_list.append(resolved_data)
            data = np.vstack(resolved_list)
        else:
            data = data_obj
    return data

# Loop para converter todos os arquivos para cada nível de torque
for filename, group in bases:
    file_path = os.path.join(base_dir, filename)
    for torque in torque_levels:
        print(f"Processando arquivo: {filename}, Grupo: {group}, Torque: {torque}")
        try:
            # Carrega os dados do nível de torque atual
            data = load_mat_raw(file_path, group, torque, signal_key="Ia")
            print(f"Shape dos dados carregados: {data.shape}")

            # Monta o nome do arquivo npy a partir do nome da base, grupo e torque
            base_name = os.path.splitext(filename)[0]  # remove a extensão, ex.: "struct_rs_R1"
            npy_filename = f"{base_name}_{group}_{torque}.npy"
            npy_path = os.path.join(output_dir, npy_filename)

            # Salva os dados convertidos em formato .npy
            np.save(npy_path, data)
            print(f"Salvo: {npy_path}")
        except Exception as e:
            print(f"Erro ao processar {filename} para {torque}: {e}")
        finally:
            gc.collect()  # Libera memória entre as conversões
