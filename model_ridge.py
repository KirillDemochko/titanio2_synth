import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')
X = df.iloc[:, :-5].copy()
y = df.iloc[:, -5:]
target_names = ['Tube_diameter_nm', 'Tube_length_um', 'Wall_thickness_nm',
                'Pore_density_pores_per_um2', 'Anatase_ratio']

X['Annealing_atmosphere'] = X['Annealing_atmosphere'].astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_pred = np.zeros((len(X_test), 5))
for i in range(5):
    print(f"\n[{i+1}/5] Training Ridge for: {target_names[i]}")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_scaled, y_train.iloc[:, i])
    y_pred[:, i] = model.predict(X_test_scaled)
    print(f"[{i+1}/5] Done!")

print(f"\n{'='*60}")
print("METRICS FOR EACH TARGET:")
print(f"{'='*60}")
for i in range(5):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{target_names[i]:30s} | MAE: {mae:7.4f}, RMSE: {rmse:7.4f}, R2: {r2:.4f}")

mae_avg = np.mean([mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]) for i in range(5)])
rmse_avg = np.mean([np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])) for i in range(5)])
r2_avg = np.mean([r2_score(y_test.iloc[:, i], y_pred[:, i]) for i in range(5)])
print(f"AVERAGE      | MAE: {mae_avg:7.4f}, RMSE: {rmse_avg:7.4f}, R2: {r2_avg:.4f}")

print("\nShowing plots for each target...")
for i in range(5):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.7, edgecolors='k', linewidth=0.5)
    plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
             [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'r--', linewidth=2)
    plt.xlabel("True value", fontsize=11)
    plt.ylabel("Predicted value", fontsize=11)
    plt.title(f"{target_names[i]}", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

pd.DataFrame(y_pred, columns=target_names).to_csv("ridge_predictions.csv", index=False)
print("\nPredictions saved to ridge_predictions.csv")