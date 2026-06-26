import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# --- Load Data ---
df = pd.read_csv('heart.csv')

# ============================================
# 👇 DATA CLEANING CODE GOES HERE 👇
# ============================================
df = df.drop_duplicates()          # remove duplicate rows
df = df[df['thal'] != 0]           # remove invalid category
df = df[df['ca'] != 4]             # remove invalid category

def remove_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[col] >= Q1 - 1.5*IQR) & (data[col] <= Q3 + 1.5*IQR)]

for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
    df = remove_outliers(df, col)
# ============================================
# 👆 END OF DATA CLEANING 👆
# ============================================

# --- Split Features & Target ---
X = df.drop('target', axis=1)
y = df['target']

# --- Scale ---
scaler = StandardScaler()
cols_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Train Accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# --- Save Model & Scaler ---
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Saved model.pkl and scaler.pkl")