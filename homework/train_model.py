"""Build, deploy and access a model using scikit-learn"""

import pickle
import pandas as pd  #  type: ignore
from pathlib import Path

from sklearn.linear_model import LinearRegression  # type: ignore

BASE = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE / "files" / "input" / "house_data.csv", sep=",")

features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

target = df[["price"]]

estimator = LinearRegression()
estimator.fit(features, target)

with open(BASE / "homework" / "house_predictor.pkl", "wb") as file:
    pickle.dump(estimator, file)


# ----------------
