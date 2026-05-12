import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/data.csv")

# Converter data
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df = df.drop("date", axis=1)

# Remover colunas ruins
df = df.drop(["street", "country"], axis=1)

# Categóricas
df = pd.get_dummies(df, columns=["city"], drop_first=True)
df = pd.get_dummies(df, columns=["statezip"], drop_first=True)

# Remover outliers
df = df[df["price"] < df["price"].quantile(0.95)]

df = df[df["price"] > 0]

# Features e target (log)
X = df.drop("price", axis=1)
y = np.log(df["price"])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsão
y_pred = model.predict(X_test)

# Voltar para escala real
y_pred = np.exp(y_pred)
y_test_exp = np.exp(y_test)

# Métricas
mae = mean_absolute_error(y_test_exp, y_pred)
r2 = r2_score(y_test_exp, y_pred)

print("MAE:", mae)
print("R2:", r2)

# Erro percentual
mean_price = df["price"].mean()
mae_percent = mae / mean_price

print("Erro percentual:", mae_percent)



