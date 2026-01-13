# training_pso.py
import numpy as np
import pandas as pd
import joblib
import pyswarms as ps
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ===============================
# 1. Load dataset
# ===============================
csv_filename = 'delhi_metro_updated.csv'  # make sure this CSV is in Colab folder
data = pd.read_csv(csv_filename)
data = data.head(1000)  # optional: limit rows

# Features and target
feature_names = ['Distance_km', 'Fare', 'Cost_per_passenger']
X = data[feature_names].values
y = data['Passengers'].values.reshape(-1, 1)

# ===============================
# 2. Scale features
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ===============================
# 3. Define PSO fitness function
# ===============================
def fitness_function(params):
    n_particles = params.shape[0]
    mse_vals = np.zeros(n_particles)

    for i in range(n_particles):
        weights = params[i, :-1].reshape(-1, 1)  # w1, w2, w3
        bias = params[i, -1]
        y_pred = X_train @ weights + bias
        mse_vals[i] = mean_squared_error(y_train, y_pred)

    return mse_vals

# ===============================
# 4. Run PSO
# ===============================
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
bounds = (np.array([0, 0, 0, -100]), np.array([5, 5, 5, 100]))  # weights + bias

optimizer = ps.single.GlobalBestPSO(
    n_particles=30, dimensions=4, options=options, bounds=bounds
)

best_cost, best_pos = optimizer.optimize(fitness_function, iters=100)

# ===============================
# 5. Extract weights and bias
# ===============================
best_weights = best_pos[:-1].reshape(-1, 1)
best_bias = best_pos[-1]

# ===============================
# 6. Evaluate on test set
# ===============================
y_test_pred = X_test @ best_weights + best_bias
y_test_pred = np.clip(y_test_pred, 0, None)  # prevent negative predictions
mse_test = mean_squared_error(y_test, y_test_pred)

print("Best Weights:", best_weights.flatten())
print("Best Bias:", best_bias)
print("Test MSE:", mse_test)

# ===============================
# 7. Save model and scaler
# ===============================
joblib.dump({"model_type": "Multi-Objective PSO Optimized Linear Regression", "weights": best_weights, "bias": best_bias, "feature_names": feature_names}, "pso_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved successfully!")

# ===============================
# 8. Optional: Demo prediction
# ===============================
X_demo = np.array([[5.0, 20.0, 15.0]])  # Distance, Fare, Cost per Passenger
X_demo_scaled = scaler.transform(X_demo)
y_demo = X_demo_scaled @ best_weights + best_bias
y_demo = max(0, y_demo.item())
print("Demo Estimated Passengers:", y_demo)

y_max_demo = max(0, y_demo * 1.1)
print("Demo Max Passengers (+10% margin):", y_max_demo)
