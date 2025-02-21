# ====================
# 1. Librerías
# ====================

import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# ====================
# 2. Cargar los datos preprocesados
# ====================

X_train, X_test, Y_train, Y_test = joblib.load('../../data/processed/split_data.pkl')

# ====================
# 3. Creación Modelo
# ====================

modelo_nb = GaussianNB()
modelo_nb.fit(X_train, Y_train)
y_pred_nb = modelo_nb.predict(X_test)

# ====================
# 4. Evaluación del Modelo
# ====================

accuracy_nb = accuracy_score(Y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(Y_test, y_pred_nb)
class_report_nb = classification_report(Y_test, y_pred_nb)

print(f'Naive Bayes Accuracy: {accuracy_nb}')
print('Naive Bayes Confusion Matrix:')
print(conf_matrix_nb)
print('Naive Bayes Classification Report:')
print(class_report_nb)

# ====================
# 5. Visualización
# ====================

# Crear la matriz de confusión que compare las predicciones con los valores reales para Naive Bayes
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['No Exited', 'Exited'], yticklabels=['No Exited', 'Exited'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes Confusion Matrix')
plt.show()

# Calcular las probabilidades de predicción para Naive Bayes
y_pred_proba_nb = modelo_nb.predict_proba(X_test)[:, 1]

# Calcular la curva ROC para Naive Bayes
fpr_nb, tpr_nb, thresholds_nb = roc_curve(Y_test, y_pred_proba_nb)

# Calcular el AUC para Naive Bayes
roc_auc_nb = roc_auc_score(Y_test, y_pred_proba_nb)

# Visualizar la curva ROC para Naive Bayes
plt.figure(figsize=(10,7))
plt.plot(fpr_nb, tpr_nb, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_nb:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
