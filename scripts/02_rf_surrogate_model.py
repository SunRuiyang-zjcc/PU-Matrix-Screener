"""
Module 02: Random Forest Surrogate QSAR Modeling
Trains models mapping molecular topology/embeddings to Ionization Efficiency & Crystal Size.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


class MatrixSurrogateModel:
    def __init__(self):
        # Dual-target models as specified in the NSFC proposal
        self.rf_ionization = RandomForestRegressor(n_estimators=200, random_state=42)
        self.rf_crystal_size = RandomForestRegressor(n_estimators=200, random_state=42)

    def train(self, X_embeddings, y_ionization, y_crystal):
        print("🚀 Training Random Forest Surrogates on multi-dimensional features...")

        # Train Ionization Efficiency Model
        X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y_ionization, test_size=0.2)
        self.rf_ionization.fit(X_train, y_train)
        print(f"   ✅ Ionization Model R2 Score: {r2_score(y_test, self.rf_ionization.predict(X_test)):.3f}")

        # Train Crystal Size Model (Lower size is better for Single-Cell MSI)
        X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y_crystal, test_size=0.2)
        self.rf_crystal_size.fit(X_train, y_train)
        print(f"   ✅ Crystal Size Model R2 Score: {r2_score(y_test, self.rf_crystal_size.predict(X_test)):.3f}")

    def save_models(self, path_prefix="rf_model"):
        joblib.dump(self.rf_ionization, f"{path_prefix}_ionization.pkl")
        joblib.dump(self.rf_crystal_size, f"{path_prefix}_crystal.pkl")
        print("💾 Models successfully saved for RL Environment.")


if __name__ == "__main__":
    # Placeholder for running the training pipeline
    print("Step 02: Initiating QSAR Surrogate Training...")