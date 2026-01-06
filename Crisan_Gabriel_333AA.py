import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Ignorare warning-uri
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

# =============================================================================
# 1. Procesarea datelor
# =============================================================================
print("--- 1. Procesarea datelor ---\n")

# Incarcarea setului de date
df = pd.read_csv('salary.csv')
print("Setul de date a fost incarcat cu succes")

# Curatarea datelor (eliminarea valorilor lipsa)
df = df.dropna()

# Eliminarea coloanelor inutile
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# -----------------------------------------------------------------------------
# Feature engineering: Extragere senioritate si departament din Job Title
# -----------------------------------------------------------------------------
def process_job_title(title):
    title = str(title).lower()

    # 1. Determinare senioritate
    if any(x in title for x in ['junior', 'entry', 'intern', 'jr']):
        seniority = 0
    elif any(x in title for x in ['senior', 'sr', 'principal', 'lead']):
        seniority = 2
    elif any(x in title for x in ['director', 'chief', 'vp', 'ceo', 'cfo', 'cto', 'president', 'head']):
        seniority = 4
    elif any(x in title for x in ['manager']):
        seniority = 3
    else:
        seniority = 1

    # 2. Determinare departament
    if any(x in title for x in ['marketing', 'social media', 'content', 'copywriter', 'creative']):
        dept = 'Marketing'
    elif any(x in title for x in ['finance', 'financial', 'accountant', 'advisor', 'accounting']):
        dept = 'Finance'
    elif any(x in title for x in ['hr', 'human resources', 'recruiter', 'training']):
        dept = 'HR'
    elif any(x in title for x in ['sales', 'account', 'business development']):
        dept = 'Sales'
    elif any(x in title for x in ['software', 'developer', 'engineer', 'web', 'tech', 'net', 'it']):
        dept = 'IT/Engineering'
    elif any(x in title for x in ['data', 'scientist', 'intelligence', 'analytics']):
        dept = 'Data Science'
    elif any(x in title for x in ['product', 'design', 'ux', 'ui']):
        dept = 'Product/Design'
    elif any(x in title for x in ['operations', 'supply chain', 'logistics']):
        dept = 'Operations'
    elif any(x in title for x in ['customer', 'support', 'service']):
        dept = 'Customer Service'
    elif any(x in title for x in ['project', 'program']):
        dept = 'Project Management'
    elif any(x in title for x in ['research', 'scientific']):
        dept = 'Research'
    else:
        dept = 'Other/Admin'

    return pd.Series([seniority, dept])


print("Se proceseaza 'Job Title'...")
df[['Seniority', 'Department']] = df['Job Title'].apply(process_job_title)

# Codificare departament
le_dept = LabelEncoder()
df['Department'] = le_dept.fit_transform(df['Department'])

# Eliminare coloana originala
df = df.drop(columns=['Job Title'])

# Transformari standard
education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df['Education Level'] = df['Education Level'].map(education_mapping)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Transformarea la scara logaritmica a tintei
df['Salary_Log'] = np.log1p(df['Salary'])

# Definirea X si y
X = df.drop(columns=['Salary', 'Salary_Log'])
y = df['Salary_Log']

# Impartirea setului de date
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Date procesate. Dimensiuni Train: {X_train.shape}, Test: {X_test.shape}")

results_list = []

def evaluate_and_plot(model, X_test, y_test, model_name):
    # Predictie pe scara log
    y_pred_log = model.predict(X_test)

    # Metrici logaritmice
    mae_log = mean_absolute_error(y_test, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    r2_log = r2_score(y_test, y_pred_log)

    # Transformare inversa (reala)
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred_log)

    # Metrici reale
    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    # Salvare rezultate
    results_list.append({
        'Model': model_name,
        'MAE (Log)': mae_log,
        'RMSE (Log)': rmse_log,
        'MAE (Real)': mae_real,
        'RMSE (Real)': rmse_real,
        'R2 Score': r2_log
    })

    # Afisare in consola
    print(f"\nPerformanta {model_name}:")
    print(f"  MAE  (Log):  {mae_log:.4f}")
    print(f"  RMSE (Log):  {rmse_log:.4f}")
    print(f"  MAE  (Real): {mae_real:.0f}")
    print(f"  RMSE (Real): {rmse_real:.0f}")
    print(f"  R2 Score:    {r2_log:.4f}")
    print("")

    # Grafic
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_real, y_pred_real, alpha=0.7, color='blue', edgecolor='k')

    # Linia ideala
    min_val = min(y_test_real.min(), y_pred_real.min())
    max_val = max(y_test_real.max(), y_pred_real.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    plt.xlabel('Salariu real')
    plt.ylabel('Salariu prezis')
    plt.title(f'{model_name}: Real vs prezis')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. Antrenarea si evaluarea modelelor
# =============================================================================
print("\n--- 2. Antrenarea si evaluarea modelelor ---\n")

# --- Model 1: Decision Tree ---
print("Antrenare model 1: Decision Tree...")
dt_model = DecisionTreeRegressor(random_state=42, max_depth=10)
dt_model.fit(X_train, y_train)
evaluate_and_plot(dt_model, X_test, y_test, "Decision Tree")

# --- Model 2: Random Forest ---
print("Antrenare model 2: Random Forest (Grid Search)...")
rf = RandomForestRegressor(random_state=42)
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
evaluate_and_plot(best_rf_model, X_test, y_test, "Random Forest")

# --- Model 3: Gradient Boosting ---
print("Antrenare model 3: Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
evaluate_and_plot(gb_model, X_test, y_test, "Gradient Boosting")

# =============================================================================
# 3. Analiza comparativa finala
# =============================================================================
print("\n--- 3. Analiza comparativa finala ---\n")

comparison_df = pd.DataFrame(results_list)
comparison_df = comparison_df.sort_values(by='R2 Score', ascending=False)

print("Tabel comparativ:")
print(comparison_df.to_string(index=False))

# Bar chart pentru R2 Score (0.8 - 1)
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=comparison_df, x='Model', y='R2 Score', palette='viridis')
plt.ylim(0.8, 1.0)
plt.title('Comparatie performanta modele (R2 Score)')
plt.ylabel('R2 Score')

# Adaugarea valorilor R2 Score pe fiecare bara
for p in ax.patches:
    ax.annotate(f'{p.get_height():.4f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points',
                fontweight='bold',
                color='black')

plt.show()