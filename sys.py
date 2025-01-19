import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QFileDialog, QWidget
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class AdaptiveLearningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive Learning AI")
        self.setGeometry(100, 100, 500, 300)

        # Initial Layout
        self.layout = QVBoxLayout()
        self.label = QLabel("Load data to start adaptive learning!")
        self.layout.addWidget(self.label)
        
        self.load_data_btn = QPushButton("Load Data")
        self.load_data_btn.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_data_btn)

        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.train_model)
        self.train_model_btn.setEnabled(False)
        self.layout.addWidget(self.train_model_btn)

        self.accuracy_label = QLabel("")
        self.layout.addWidget(self.accuracy_label)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        # Attributes for model and data
        self.data = None
        self.target = None
        self.model = SGDClassifier(max_iter=1000, tol=1e-3)
        self.scaler = StandardScaler()

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load CSV File", "", "CSV Files (*.csv)")
        if file_path:
            import pandas as pd
            dataset = pd.read_csv(file_path)
            self.data = dataset.iloc[:, :-1].values
            self.target = dataset.iloc[:, -1].values
            self.train_model_btn.setEnabled(True)
            self.label.setText("Data loaded successfully!")

    def train_model(self):
        if self.data is not None and self.target is not None:
            X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            self.model.partial_fit(X_train, y_train, np.unique(y_train))
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.accuracy_label.setText(f"Model trained! Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AdaptiveLearningApp()
    window.show()
    sys.exit(app.exec_())
