# ====================
# 1. Librerías
# ====================

import joblib
from sklearn.linear_model import LogisticRegression
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

modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_train, Y_train)
y_pred_logistico = modelo_logistico.predict(X_test)

# ====================
# 4. Evaluación del Modelo
# ====================

accuracy_logistico = accuracy_score(Y_test, y_pred_logistico)
conf_matrix_logistico = confusion_matrix(Y_test, y_pred_logistico)
class_report_logistico = classification_report(Y_test, y_pred_logistico)

print(f'Logistic Regression Accuracy: {accuracy_logistico}')
print('Logistic Regression Confusion Matrix:')
print(conf_matrix_logistico)
print('Logistic Regression Classification Report:')
print(class_report_logistico)

# ====================
# 5. Visualización
# ====================

# Crear la matriz de confusión que compare las predicciones con los valores reales para Regresión Logística
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_logistico, annot=True, fmt='d', cmap='Blues', xticklabels=['No Exited', 'Exited'], yticklabels=['No Exited', 'Exited'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# Calcular las probabilidades de predicción para Regresión Logística
y_pred_proba_logistico = modelo_logistico.predict_proba(X_test)[:, 1]

# Calcular la curva ROC para Regresión Logística
fpr_logistico, tpr_logistico, thresholds_logistico = roc_curve(Y_test, y_pred_proba_logistico)

# Calcular el AUC para Regresión Logística
roc_auc_logistico = roc_auc_score(Y_test, y_pred_proba_logistico)

# Visualizar la curva ROC para Regresión Logística
plt.figure(figsize=(10,7))
plt.plot(fpr_logistico, tpr_logistico, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_logistico:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
