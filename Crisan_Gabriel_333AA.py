import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configurare stil grafice
sns.set(style="whitegrid")

# =============================================================================
# 1. Procesarea datelor
# =============================================================================
print("--- 1. Procesarea datelor ---\n")

# Încărcarea setului de date
try:
    df = pd.read_csv('salary.csv')
    print("Setul de date a fost încărcat cu succes.")
except FileNotFoundError:
    print("Eroare: Fișierul 'salary.csv' nu a fost găsit.")
    exit()

# Curățarea datelor (eliminare valori lipsă)
df = df.dropna()

# Eliminarea coloanelor inutile
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Transformarea variabilelor categorice
# Mapare manuală pentru Education Level (ordinal)
education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df['Education Level'] = df['Education Level'].map(education_mapping)

# Mapare pentru Gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Label Encoding pentru Job Title
le_job = LabelEncoder()
df['Job Title'] = le_job.fit_transform(df['Job Title'])

# Definirea X și y
X = df.drop(columns=['Salary'])
y = df['Salary']

# Împărțirea setului de date (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Date procesate. Dimensiuni Train: {X_train.shape}, Test: {X_test.shape}")

# Listă pentru a stoca rezultatele comparative
results_list = []


def evaluate_and_plot(model, X_test, y_test, model_name):
    """
    Funcție care face predicții, calculează metrici, le salvează și
    generează un grafic Actual vs Predicted.
    """
    # Predicție
    y_pred = model.predict(X_test)

    # Calcul metrici
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Salvare rezultate pentru comparație finală
    results_list.append({
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2 Score': r2
    })

    print(f"\nPerformanță {model_name}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2: {r2:.4f}")

    # Generare Grafic: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolor='k')

    # Linia ideală (unde y_test == y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicție Perfectă')

    plt.xlabel('Salariu Real')
    plt.ylabel('Salariu Predis')
    plt.title(f'{model_name}: Real vs Predis')
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 2. Antrenarea și evaluarea modelelor
# =============================================================================
print("\n--- 2. Antrenarea și evaluarea modelelor ---\n")

# --- Model 1: Decision Tree ---
print("Antrenare Model 1: Decision Tree...")
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
evaluate_and_plot(dt_model, X_test, y_test, "Decision Tree")

# --- Model 2: Random Forest (cu optimizare) ---
print("Antrenare Model 2: Random Forest (Grid Search)...")
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, scoring='r2')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
evaluate_and_plot(best_rf_model, X_test, y_test, "Random Forest")

# --- Model 3: Gradient Boosting ---
print("Antrenare Model 3: Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
evaluate_and_plot(gb_model, X_test, y_test, "Gradient Boosting")

# =============================================================================
# 3. Analiza Comparativă Finală
# =============================================================================
print("\n--- 3. Analiza Comparativă Finală ---\n")

# Creare DataFrame cu rezultate
comparison_df = pd.DataFrame(results_list)
comparison_df = comparison_df.sort_values(by='R2 Score', ascending=False)

print("Tabel Comparativ:")
print(comparison_df)

# Vizualizare Comparativă (Bar Chart pentru R2 Score)
plt.figure(figsize=(10, 6))
sns.barplot(data=comparison_df, x='Model', y='R2 Score', palette='viridis')
plt.ylim(0, 1)  # R2 este max 1
plt.title('Comparație Performanță Modele (R2 Score)')
plt.ylabel('R2 Score (Mai mare e mai bine)')
plt.show()