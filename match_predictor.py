import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

data = {
    "tech_score": [40, 60, 80, 90],
    "experience": [1, 2, 3, 4],
    "win": [0, 0, 1, 1]
}

df = pd.DataFrame(data)
X = df[["tech_score", "experience"]]
y = df["win"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "models/match_model.pkl")