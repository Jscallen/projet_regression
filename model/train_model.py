from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib, json
from pathlib import Path

def main():
    Path("model").mkdir(exist_ok=True)
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns=["MedHouseVal"])
    y = data.frame["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.3f} | R²: {r2:.3f}")

    joblib.dump({"model": pipe, "feature_names": list(X.columns)}, "model/model.pkl")
    Path("model/metrics.json").write_text(json.dumps({"rmse": rmse, "r2": r2}, indent=2))
    print("Modèle entraîné et sauvegardé !")

if __name__ == "__main__":
    main()
