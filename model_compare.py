import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

target_names = ['Tube_diameter_nm', 'Tube_length_um', 'Wall_thickness_nm',
                'Pore_density_pores_per_um2', 'Anatase_ratio']
models = ['CatBoost', 'RF', 'GPR', 'MLP', 'Ridge']

df = pd.read_csv('dataset.csv')
X = df.iloc[:, :-5].copy()
y = df.iloc[:, -5:]
_, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}
for name in models:
    y_pred = pd.read_csv(f'{name.lower()}_predictions.csv')
    results[name] = {
        'MAE': [mean_absolute_error(y_test.iloc[:, i], y_pred.iloc[:, i]) for i in range(5)],
        'RMSE': [np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred.iloc[:, i])) for i in range(5)],
        'R2': [r2_score(y_test.iloc[:, i], y_pred.iloc[:, i]) for i in range(5)]
    }

print("MODEL COMPARISON BY TARGET")
for i, target in enumerate(target_names):
    print(f"\n{target}:")
    print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    for name in models:
        mae = results[name]['MAE'][i]
        rmse = results[name]['RMSE'][i]
        r2 = results[name]['R2'][i]
        print(f"{name:<12} {mae:8.4f} {rmse:8.4f} {r2:8.4f}")

print("BEST MODEL PER TARGET (by R²)")
best_per_target = {}
for i, target in enumerate(target_names):
    r2_vals = [results[name]['R2'][i] for name in models]
    best_idx = np.argmax(r2_vals)
    best_model = models[best_idx]
    best_per_target[target] = best_model
    print(f"{target:30s} → {best_model:10s} (R²={r2_vals[best_idx]:.4f})")

print("OPTIMAL COMBINATION METRICS")
opt_mae, opt_rmse, opt_r2 = [], [], []
for i, target in enumerate(target_names):
    best = best_per_target[target]
    opt_mae.append(results[best]['MAE'][i])
    opt_rmse.append(results[best]['RMSE'][i])
    opt_r2.append(results[best]['R2'][i])
    print(f"{target:30s} | MAE: {opt_mae[-1]:7.4f}, RMSE: {opt_rmse[-1]:7.4f}, R2: {opt_r2[-1]:.4f}")
print(f"{'AVERAGE':30s} | MAE: {np.mean(opt_mae):7.4f}, RMSE: {np.mean(opt_rmse):7.4f}, R2: {np.mean(opt_r2):.4f}")


plt.figure(figsize=(10, 6))
x = np.arange(5)
width = 0.15
for j, name in enumerate(models):
    plt.bar(x + j*width, [results[name]['R2'][i] for i in range(5)], width, label=name)
plt.xticks(x, target_names, rotation=45, ha='right')
plt.ylabel('R²')
plt.title('R² by Model and Target')
plt.legend(fontsize=8)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.barh(target_names, opt_r2, color='#2E86AB')
plt.xlabel('R²')
plt.title('Optimal Combination: R² per Target')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
for i, target in enumerate(target_names):
    best = best_per_target[target]
    y_pred = pd.read_csv(f'{best.lower()}_predictions.csv')
    plt.subplot(1, 5, i+1)
    plt.scatter(y_test.iloc[:, i], y_pred.iloc[:, i], alpha=0.6, s=15)
    plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
             [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'r--', linewidth=1)
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title(f'{target}\n({best})', fontsize=9)
    plt.grid(alpha=0.3)
plt.suptitle('Optimal Combination: Predictions vs True', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

