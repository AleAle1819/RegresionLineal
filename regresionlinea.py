import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paso 1: Cargar el dataset ---
df = pd.read_csv(r"C:\Users\Alex\Desktop\OctavoAle\Machine Learning\dataset_correos.csv")
print("=== Vista previa del dataset ===")
print(df.head())
print("\n=== Información general ===")
print(df.info())
print("\n=== Distribución de la variable objetivo ===")
print(df['Clasificación'].value_counts())

df['Clasificación_num'] = df['Clasificación'].map({'ham': 0, 'spam': 1})


# Fecha, hora del día y Día de la semana
df['Fecha'] = pd.to_datetime(df['Fecha'])
df['Hora_del_Dia'] = df['Fecha'].dt.hour
df['Dia_de_la_Semana'] = df['Fecha'].dt.dayofweek  # 0=lunes, 6=domingo

# dominio del correo
df['Dominio'] = df['Correo_Origen'].str.split('@').str[1]
dominio_to_id = {dom: idx for idx, dom in enumerate(df['Dominio'].unique())}
df['Dominio_ID'] = df['Dominio'].map(dominio_to_id)

# palabras clave spam
palabras_spam = ["urgente", "oferta", "promoción", "descuento", "evento", "ventas", "música", "trabajo", "reunión"]
df['Palabras_Spam_En_Asunto'] = df['Asunto'].str.lower().apply(
    lambda x: sum(1 for palabra in palabras_spam if palabra in str(x).lower())
)

df['Cantidad_Palabras_Clave'] = df['Palabras_Clave'].str.split(',').apply(
    lambda x: len(x) if isinstance(x, list) else 0
)

# hay o no hay link (bool)
df['Tiene_Link'] = df['Link'].notna() & (df['Link'] != '')
df['Tiene_Link'] = df['Tiene_Link'].astype(int)

features = [
    'Hora_del_Dia',
    'Dia_de_la_Semana',
    'Dominio_ID',
    'Palabras_Spam_En_Asunto',
    'Cantidad_Palabras_Clave',
    'Tiene_Link',
    'Mayúsculas_Excesivas',
    'Archivos_Adjuntos',
    'Longitud_Correo',
    'Errores_Ortograficos',
    'Caracteres_Especiales'  
]

X = df[features]
y = df['Clasificación_num']

print("\n=== Características seleccionadas ===")
print(X.head())
print(f"\nForma de X: {X.shape}")
print(f"Forma de y: {y.shape}")

# Verificar valores nulos
print("\n=== Valores nulos en X ===")
print(X.isnull().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train test 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Crear e instanciar el modelo con balance de clases 
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# umbral
y_proba = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, (y_proba >= t).astype(int)) for t in thresholds]

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"\n🌟 Mejor umbral: {best_threshold:.2f}")
print(f"🌟 Mejor F1-Score: {best_f1:.4f}")

#F1-Score vs Umbral ---
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label='F1-Score', marker='o', linewidth=2)
plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Mejor Umbral: {best_threshold:.2f}', linewidth=2)
plt.xlabel('Umbral de Probabilidad')
plt.ylabel('F1-Score')
plt.title('F1-Score vs Umbral de Probabilidad (Con Balance de Clases y Escalado)')
plt.legend()
plt.grid(True, alpha=0.7)
plt.show()

# predicciones con el mejor umbral
y_pred_best = (y_proba >= best_threshold).astype(int)

# matriz de confusión
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
            annot_kws={"size": 14})
plt.title('Matriz de Confusión (Mejor Umbral)', fontsize=15)
plt.xlabel('Predicción', fontsize=13)
plt.ylabel('Valor Real', fontsize=13)
plt.show()

# reporte de clasificación
print("\n Reporte de Clasificación (Mejor Umbral):")
print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

# importancia de características
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\n Importancia de las Características:")
print(feature_importance.to_string(index=False))

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='Coefficient', y='Feature', palette='viridis')
plt.title('Importancia de las Características (Coeficientes de Regresión Logística)', fontsize=15)
plt.xlabel('Coeficiente', fontsize=13)
plt.ylabel('Característica', fontsize=13)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# matriz de correlación
corr_df = X.copy()
corr_df['Clasificación_num'] = y
correlation_matrix = corr_df.corr()

# Correlación con la variable objetivo
target_corr = correlation_matrix['Clasificación_num'].drop('Clasificación_num')
print("\n Correlación con la variable objetivo (Clasificación_num):")
print(target_corr.sort_values(key=abs, ascending=False).to_string())

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, annot_kws={"size": 11})
plt.title('Mapa de Calor de Correlación', fontsize=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()