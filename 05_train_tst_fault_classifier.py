tfms = [None, TSClassification()]
# Batch transform: padronização (standardize)
batch_tfms = TSStandardize()

# Cria o learner. arquitetura TST (Time Series Transformer)
learn = TSClassifier(X=X_all, y=y_all, splits=splits, arch=TST, tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy)

# Treinamento com o método fit_one_cycle (10 épocas e lr_max definido em 1e-3)
learn.fit_one_cycle(n_epoch=10, lr_max=1e-3)

#Salva o modelo treinado
PATH = Path('/content/drive/MyDrive/Mestrado/Modelos/ArchTST/TST.pkl')
PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)

#Gerar uma interpretação dos resultados do modelo de classificação treinado e visualizar a matriz de confusão
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(normalize=True)

# Carregar o modelo treinado
PATH = Path('/content/drive/MyDrive/Mestrado/Modelos/ArchTST/TST.pkl')
learn_gpu = load_learner(PATH, cpu=False)

# Gerar as previsões para o conjunto de teste
probas, _, preds = learn_gpu.get_X_preds(X_all[splits[1]])
print("Probabilidades (dados originais):")
print(probas)
print("Predições (dados originais):")
print(preds)

import sklearn.metrics as skm

# Calcula a acurácia apenas para o conjunto de teste dos dados originais
acc_orig = skm.accuracy_score(y_all[splits[1]], preds)
print("Acurácia (dados originais):", acc_orig)

import matplotlib.pyplot as plt
import sklearn.metrics as skm

# Converte para inteiros, se necessário
y_true = y_all[splits[1]].astype(int)
preds_int = preds.astype(int)

# Cria a figura
fig, ax = plt.subplots(figsize=(10, 8))
cm_display = skm.ConfusionMatrixDisplay.from_predictions(
    y_true, preds_int,
    normalize='true',
    cmap='Blues',
    ax=ax
)

# Aumenta o tamanho das fontes
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.xlabel("Classe Predita", fontsize=16)
plt.ylabel("Classe Verdadeira", fontsize=16)
plt.title("Matriz de Confusão Normalizada", fontsize=18)

# Salva com alta resolução Google Drive
plt.savefig("/content/drive/MyDrive/Mestrado/matriz_confusaoTST.png", dpi=300, bbox_inches='tight')
plt.show()

#F1-Score, Precisão e Revocação
f1 = skm.f1_score(y_true, preds_int, average='weighted')
precision = skm.precision_score(y_true, preds_int, average='weighted')
recall = skm.recall_score(y_true, preds_int, average='weighted')

print("F1-Score (weighted):", f1)
print("Precisão (weighted):", precision)
print("Revocação (weighted):", recall)

#Balanced Accuracy e Cohen's Kappa
balanced_acc = skm.balanced_accuracy_score(y_true, preds_int)
kappa = skm.cohen_kappa_score(y_true, preds_int)

print("Balanced Accuracy:", balanced_acc)
print("Cohen’s Kappa:", kappa)

#Relatório completo (Classification Report)
print("Relatório de Classificação:\n")
print(skm.classification_report(y_true, preds_int))
