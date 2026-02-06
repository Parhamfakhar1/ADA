import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class ModelManager:
    def __init__(self):
        self.model = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.classes_ = None
        self.accuracy_history = []

    def fit(self, X_train, y_train, X_test, y_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.classes_ = np.unique(y_train)
        self.model.partial_fit(X_train_scaled, y_train, classes=self.classes_)

        y_pred = self.model.predict(X_test_scaled)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)
        self.accuracy_history.append(acc)
        self.is_fitted = True
        return acc

    def update_with_sample(self, x_new, y_new):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before updating.")
        x_new = np.array(x_new).reshape(1, -1)
        x_new_scaled = self.scaler.transform(x_new)
        self.model.partial_fit(x_new_scaled, [y_new])
        # Note: We don't compute accuracy here without test set
        return True

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not trained yet.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path="model.pkl", scaler_path="scaler.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_fitted = True
        self.classes_ = self.model.classes_