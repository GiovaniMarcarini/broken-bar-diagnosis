# Parâmetros de segmentação
seq_length = 100  # Comprimento de cada segmento
step = 10         # Passo entre inícios de segmentos

# Lista dos níveis de torque
torque_levels = ["torque05", "torque10", "torque15", "torque20",
                 "torque25", "torque30", "torque35", "torque40"]

# Diretório onde os arquivos .npy convertidos estão salvos
npy_dir = '/content/drive/MyDrive/Mestrado/Converted/'

# Diretório para salvar os splits intermediários
temp_dir = '/content/drive/MyDrive/Mestrado/TempSplits/'
os.makedirs(temp_dir, exist_ok=True)

def load_npy_segments(file_path, segment_length=seq_length, step_size=step):
    """
    Carrega um arquivo .npy com mapeamento de memória e segmenta o sinal.

    Assume que o sinal tem shape (n_channels, n_samples) e retorna
    um array com shape (n_segments, n_channels, segment_length).
    """
    # Carrega com mmap_mode para minimizar o uso de RAM
    data = np.load(file_path, mmap_mode='c')
    n_channels, n_samples = data.shape
    segments = []
    for start in range(0, n_samples - segment_length + 1, step_size):
        segment = data[:, start:start + segment_length]
        segments.append(segment)
    # Essa conversão para np.array aloca o array final na memória
    segments = np.array(segments)
    return segments

def process_and_save_split(base_name, group, torque, label, npy_dir, temp_dir,
                           segment_length=seq_length, step_size=step,
                           test_size=0.2, random_state=42):

    npy_filename = f"{base_name}_{group}_{torque}.npy"
    file_path = os.path.join(npy_dir, npy_filename)
    print(f"Processando {npy_filename}...")

    try:
        segments = load_npy_segments(file_path, segment_length, step_size)
    except Exception as e:
        print(f"Erro ao carregar {npy_filename}: {e}")
        return

    if segments.shape[0] == 0:
        print(f"{npy_filename} não gerou segmentos.")
        return

    # Divide os segmentos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        segments, np.full(segments.shape[0], label),
        test_size=test_size, random_state=random_state)

    # Define os nomes dos arquivos para salvar os splits
    train_file = os.path.join(temp_dir, f"{base_name}_{group}_{torque}_train.npz")
    test_file  = os.path.join(temp_dir, f"{base_name}_{group}_{torque}_test.npz")

    # Se os arquivos já existirem, remove-os antes de salvar os novos dados
    if os.path.exists(train_file):
        os.remove(train_file)
    if os.path.exists(test_file):
        os.remove(test_file)

    # Salva os splits usando np.savez_compressed para economizar espaço
    np.savez_compressed(train_file, X_train=X_train, y_train=y_train)
    np.savez_compressed(test_file, X_test=X_test, y_test=y_test)
    print(f"Salvo {npy_filename}: Treino {X_train.shape[0]} | Teste {X_test.shape[0]}")

    # Libera a memória dos arrays processados
    del segments, X_train, X_test, y_train, y_test
    gc.collect()


def process_class_splits(base_name, group, torque_list, label, npy_dir, temp_dir,
                         segment_length=seq_length, step_size=step,
                         test_size=0.2, random_state=42):
    """
    Para uma classe (ex.: motor saudável "rs"), processa cada nível de torque e
    salva os splits intermediários.
    """
    for torque in torque_list:
        process_and_save_split(base_name, group, torque, label, npy_dir, temp_dir,
                               segment_length, step_size, test_size, random_state)

process_class_splits("struct_rs_R1", "rs", torque_levels, 0, npy_dir, temp_dir, seq_length, step, 0.2)
process_class_splits("struct_r1b_R1", "r1b", torque_levels, 1, npy_dir, temp_dir, seq_length, step, 0.2)
process_class_splits("struct_r2b_R1", "r2b", torque_levels, 2, npy_dir, temp_dir, seq_length, step, 0.2)
process_class_splits("struct_r3b_R1", "r3b", torque_levels, 3, npy_dir, temp_dir, seq_length, step, 0.2)
process_class_splits("struct_r4b_R1", "r4b", torque_levels, 4, npy_dir, temp_dir, seq_length, step, 0.2)
