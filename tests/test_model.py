import numpy as np
from model_manager import ModelManager
from sklearn.datasets import make_classification

def test_model_training():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=3, n_informative=3, random_state=42)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mgr = ModelManager()
    acc = mgr.fit(X_train, y_train, X_test, y_test)
    assert 0.0 <= acc <= 1.0, "Accuracy out of range"
    print(f"✅ Test passed! Accuracy: {acc:.2f}")

if __name__ == "__main__":
    test_model_training()