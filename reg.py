import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

df = pd.read_csv('ml_moscow_flats.csv')
print("Исходный размер:", df.shape)

df = df.drop_duplicates()

freq_cats = df['wallsMaterial'].value_counts()
keep_cats = freq_cats[freq_cats >= 50].index.tolist()
df = df[df['wallsMaterial'].isin(keep_cats)]
print("После фильтрации редких категорий:", df.shape)

q005 = df['price'].quantile(0.005)
q995 = df['price'].quantile(0.995)
df = df[(df['price'] >= q005) & (df['price'] <= q995)]

df['log_totalArea'] = np.log(df['totalArea'])
df['log_kitchenArea'] = np.log(df['kitchenArea'])

X = df.drop('price', axis=1)
y = df['price']
y_log = np.log(y)

categorical_features = ['wallsMaterial']
numeric_features = ['floorNumber', 'floorsTotal', 'log_totalArea', 'log_kitchenArea', 'latitude', 'longitude']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

lasso_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('lasso', LassoCV(cv=5, random_state=42, alphas=np.logspace(-5, 2, 100), max_iter=10000))
])
lasso_pipe.fit(X_train, y_train_log)
y_pred_log_lasso = lasso_pipe.predict(X_test)
y_pred_lasso = np.exp(y_pred_log_lasso)
y_test = np.exp(y_test_log)

et_pipe = Pipeline([
    ('preprocess', preprocessor),
    ('et', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])
et_pipe.fit(X_train, y_train_log)
y_pred_log_et = et_pipe.predict(X_test)
y_pred_et = np.exp(y_pred_log_et)

def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name}")
    print(f"MSE   : {mse:.2f} руб²")
    print(f"MAE   : {mae:.2f} руб")
    print(f"R²    : {r2:.4f}")

print_metrics(y_test, y_pred_lasso, "LassoCV (лог цена + лог площади + RobustScaler)")
print_metrics(y_test, y_pred_et, "ExtraTreesRegressor")

cv_lasso_mse = cross_val_score(lasso_pipe, X_train, y_train_log, cv=5,
                               scoring='neg_mean_squared_error')
cv_lasso_r2  = cross_val_score(lasso_pipe, X_train, y_train_log, cv=5,
                               scoring='r2')
print("\nКросс‑валидация (5‑fold) LassoCV (лог цены):")
print(f"MSE = {-cv_lasso_mse.mean():.4f} (+/- {cv_lasso_mse.std():.4f})")
print(f"R²  = {cv_lasso_r2.mean():.4f} (+/- {cv_lasso_r2.std():.4f})")

cv_et_mse = cross_val_score(et_pipe, X_train, y_train_log, cv=5,
                            scoring='neg_mean_squared_error')
cv_et_r2  = cross_val_score(et_pipe, X_train, y_train_log, cv=5,
                            scoring='r2')
print("\nКросс‑валидация (5‑fold) ExtraTreesRegressor:")
print(f"MSE = {-cv_et_mse.mean():.4f} (+/- {cv_et_mse.std():.4f})")
print(f"R²  = {cv_et_r2.mean():.4f} (+/- {cv_et_r2.std():.4f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(y_test, y_pred_lasso, alpha=0.5, color='cornflowerblue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_title(f'LassoCV\nR² = {r2_score(y_test, y_pred_lasso):.3f}')
axes[0].set_xlabel('Реальная цена (руб)')
axes[0].set_ylabel('Предсказанная цена (руб)')
axes[1].scatter(y_test, y_pred_et, alpha=0.5, color='salmon')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_title(f'ExtraTreesRegressor\nR² = {r2_score(y_test, y_pred_et):.3f}')
axes[1].set_xlabel('Реальная цена (руб)')
plt.tight_layout()
plt.show()

preprocessor.fit(X_train)
feature_names = (numeric_features +
                 list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
et_model = et_pipe.named_steps['et']
importances = et_model.feature_importances_

plt.figure(figsize=(10, 5))
indices = np.argsort(importances)[::-1]
plt.barh(range(len(importances)), importances[indices], color='teal')
plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
plt.xlabel('Важность признака')
plt.title('Важность признаков (ExtraTreesRegressor)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()