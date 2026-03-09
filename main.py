import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)


n_players = 300

win_rate = np.random.uniform(0.2, 0.9, n_players)                  
kills_per_match = np.random.uniform(2, 15, n_players)
damage_per_match = np.random.uniform(5000, 30000, n_players)
matches_this_week = np.random.uniform(1, 20, n_players)


true_satisfaction = (
    3.0 * win_rate +
    2.5 * (kills_per_match / 15) +
    1.5 * (damage_per_match / 30000) +
    0.5 * (matches_this_week / 20)
)
true_satisfaction = np.clip(true_satisfaction * 10, 0, 10)  


X = np.column_stack([win_rate, kills_per_match, damage_per_match, matches_this_week])


X_train, X_val, y_true_train, y_true_val = train_test_split(
    X, true_satisfaction, test_size=0.3, random_state=42
)


rage_levels = [0, 1, 2, 4, 8]  

train_mses = []
val_mses = []
coefs_list = []

for rage in rage_levels:
    
    rage_noise = np.random.normal(0, rage, len(y_true_train))
    y_train_noisy = np.clip(y_true_train + rage_noise, 0, 10)
    
    
    model = LinearRegression()
    model.fit(X_train, y_train_noisy)
    
    
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    train_mse = mean_squared_error(y_train_noisy, y_pred_train)
    val_mse = mean_squared_error(y_true_val, y_pred_val)  
    
    train_mses.append(train_mse)
    val_mses.append(val_mse)
    coefs_list.append(model.coef_)
    
    print(f"Rage noise std {rage} | Train MSE: {train_mse:.2f} | Val MSE: {val_mse:.2f}")
    print(f"   Coefs: win_rate={model.coef_[0]:5.2f}, kills={model.coef_[1]:5.2f}, damage={model.coef_[2]:5.2f}, matches={model.coef_[3]:5.2f}")


plt.figure(figsize=(8, 5))
plt.plot(rage_levels, train_mses, 'o-', color='blue', label='Train MSE (noisy)')
plt.plot(rage_levels, val_mses, 'o-', color='orange', label='Val MSE (clean)')
plt.xlabel('Rage survey noise (std)')
plt.ylabel('MSE')
plt.title('When Toxic Reviews Poison Satisfaction Prediction')
plt.legend()
plt.grid(True)
plt.show()


coefs = np.array(coefs_list)
feature_names = ['Win Rate', 'Kills/Match', 'Damage/Match', 'Matches/Week']
plt.figure(figsize=(10, 6))
for i, name in enumerate(feature_names):
    plt.plot(rage_levels, coefs[:, i], 'o-', label=name)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlabel('Rage noise std')
plt.ylabel('Learned coefficient')
plt.title('How Rage Noise Makes Coefficients Unstable')
plt.legend()
plt.grid(True)
plt.show()