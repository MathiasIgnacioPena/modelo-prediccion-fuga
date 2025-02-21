# ====================
# 1. Librerías
# ====================

import joblib
from sklearn.ensemble import RandomForestClassifier
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

modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, Y_train)
y_pred_rf = modelo_rf.predict(X_test)

# ====================
# 4. Evaluación del Modelo
# ====================

accuracy_rf = accuracy_score(Y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(Y_test, y_pred_rf)
class_report_rf = classification_report(Y_test, y_pred_rf)

print(f'Random Forest Accuracy: {accuracy_rf}')
print('Random Forest Confusion Matrix:')
print(conf_matrix_rf)
print('Random Forest Classification Report:')
print(class_report_rf)

# ====================
# 5. Visualización
# ====================

# Crear la matriz de confusión que compare las predicciones con los valores reales para Random Forest
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['No Exited', 'Exited'], yticklabels=['No Exited', 'Exited'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Calcular las probabilidades de predicción para Random Forest
y_pred_proba_rf = modelo_rf.predict_proba(X_test)[:, 1]

# Calcular la curva ROC para Random Forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(Y_test, y_pred_proba_rf)

# Calcular el AUC para Random Forest
roc_auc_rf = roc_auc_score(Y_test, y_pred_proba_rf)

# Visualizar la curva ROC para Random Forest
plt.figure(figsize=(10,7))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
