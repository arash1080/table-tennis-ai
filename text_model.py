import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/iranian_players.csv")

X = df.drop("skill_level", axis=1)
y = df["skill_level"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "models/text_model.pkl")