import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('ml_moscow_flats.csv')
df = df.drop_duplicates()
freq_cats = df['wallsMaterial'].value_counts()
keep_cats = freq_cats[freq_cats >= 50].index
df = df[df['wallsMaterial'].isin(keep_cats)]
q005, q995 = df['price'].quantile([0.005, 0.995])
df = df[(df['price'] >= q005) & (df['price'] <= q995)]

X_clust = df.drop('price', axis=1).copy()
X_clust = pd.get_dummies(X_clust, columns=['wallsMaterial'], drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)

inertia = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

diff = np.diff(inertia)
optimal_k = K_range[np.argmin(diff) + 1]

plt.figure(figsize=(7,5))
plt.plot(K_range, inertia, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'K = {optimal_k}')
plt.xlabel('Количество кластеров (K)')
plt.ylabel('Инерция')
plt.title('Метод локтя')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

print("\nХарактеристики кластеров")
for i in range(optimal_k):
    data = df[clusters == i]
    if len(data) > 0:
        print(f"\nКластер {i+1}:")
        print(f"  Квартир: {len(data)}")
        print(f"  Средняя цена: {data['price'].mean():,.0f} руб")
        print(f"  Средняя общая площадь: {data['totalArea'].mean():.1f} кв.м")
        print(f"  Средний этаж: {data['floorNumber'].mean():.1f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title(f'Кластеры KMeans (K={optimal_k})')
plt.grid(alpha=0.3)
plt.show()