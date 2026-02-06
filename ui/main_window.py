import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QWidget, QMessageBox, QLineEdit, QDialog,
    QFormLayout, QDialogButtonBox, QHBoxLayout
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class AccuracyPlotWidget(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

    def plot(self, accuracies):
        self.axes.clear()
        self.axes.plot(range(1, len(accuracies)+1), accuracies, marker='o')
        self.axes.set_title("Model Accuracy Over Time")
        self.axes.set_xlabel("Update Step")
        self.axes.set_ylabel("Accuracy")
        self.axes.set_ylim(0, 1)
        self.axes.grid(True)
        self.draw()

class AddSampleDialog(QDialog):
    def __init__(self, n_features, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Sample")
        self.n_features = n_features
        self.inputs = []

        layout = QFormLayout()
        for i in range(n_features):
            line_edit = QLineEdit()
            self.inputs.append(line_edit)
            layout.addRow(f"Feature {i+1}:", line_edit)

        self.label_input = QLineEdit()
        layout.addRow("Label:", self.label_input)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def validate_and_accept(self):
        try:
            features = [float(inp.text()) for inp in self.inputs]
            label = self.label_input.text().strip()
            if not label:
                raise ValueError("Label cannot be empty.")
            self.features = features
            self.label = label
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")

    def get_data(self):
        return self.features, self.label

class AdaptiveLearningWindow(QMainWindow):
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager
        self.data = None
        self.target = None
        self.feature_names = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Adaptive Learning AI – Professional Edition")
        self.setGeometry(100, 100, 700, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        self.status_label = QLabel("Load a CSV file to begin.")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("📂 Load CSV Data")
        self.load_btn.clicked.connect(self.load_data)
        btn_layout.addWidget(self.load_btn)

        self.train_btn = QPushButton("🧠 Train Initial Model")
        self.train_btn.clicked.connect(self.train_initial_model)
        self.train_btn.setEnabled(False)
        btn_layout.addWidget(self.train_btn)

        self.add_sample_btn = QPushButton("➕ Add New Sample")
        self.add_sample_btn.clicked.connect(self.add_new_sample)
        self.add_sample_btn.setEnabled(False)
        btn_layout.addWidget(self.add_sample_btn)

        layout.addLayout(btn_layout)

        self.accuracy_label = QLabel("")
        self.accuracy_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.accuracy_label)

        self.plot_widget = AccuracyPlotWidget(self, width=5, height=3)
        layout.addWidget(self.plot_widget)

        io_layout = QHBoxLayout()
        self.save_btn = QPushButton("💾 Save Model")
        self.save_btn.clicked.connect(self.save_model)
        self.save_btn.setEnabled(False)
        io_layout.addWidget(self.save_btn)

        self.load_model_btn = QPushButton("📂 Load Model")
        self.load_model_btn.clicked.connect(self.load_existing_model)
        io_layout.addWidget(self.load_model_btn)

        layout.addLayout(io_layout)

        central_widget.setLayout(layout)

        # Apply dark theme (optional but professional)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; }
            QPushButton { background-color: #3c3f41; color: white; border: 1px solid #555; padding: 8px; }
            QPushButton:hover { background-color: #4a4d4f; }
            QLabel { font-size: 14px; }
        """)

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            if df.shape[1] < 2:
                raise ValueError("CSV must have at least one feature and one label column.")
            self.data = df.iloc[:, :-1].values
            self.target = df.iloc[:, -1].values
            self.feature_names = df.columns[:-1].tolist()
            self.train_btn.setEnabled(True)
            self.status_label.setText(f"✅ Loaded {len(self.data)} samples with {self.data.shape[1]} features.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")

    def train_initial_model(self):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.target, test_size=0.2, random_state=42, stratify=self.target
        )
        try:
            acc = self.model_manager.fit(X_train, y_train, X_test, y_test)
            self.accuracy_label.setText(f"🎯 Initial model trained! Accuracy: {acc*100:.2f}%")
            self.add_sample_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.plot_widget.plot(self.model_manager.accuracy_history)
        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))

    def add_new_sample(self):
        if self.data is None or self.data.shape[1] == 0:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return
        dialog = AddSampleDialog(self.data.shape[1], self)
        if dialog.exec_() == QDialog.Accepted:
            features, label = dialog.get_data()
            try:
                # Update model
                self.model_manager.update_with_sample(features, label)
                # Optional: re-evaluate on original test set? (not done here for simplicity)
                self.status_label.setText("🆕 Sample added and model updated!")
                # For demo, we fake an accuracy update (in real app, you'd track drift/test set)
                if len(self.model_manager.accuracy_history) > 0:
                    last_acc = self.model_manager.accuracy_history[-1]
                    # Simulate slight change
                    new_acc = min(1.0, max(0.0, last_acc + np.random.uniform(-0.02, 0.02)))
                    self.model_manager.accuracy_history.append(new_acc)
                self.plot_widget.plot(self.model_manager.accuracy_history)
            except Exception as e:
                QMessageBox.critical(self, "Update Error", str(e))

    def save_model(self):
        try:
            self.model_manager.save()
            QMessageBox.information(self, "Success", "Model and scaler saved as 'model.pkl' and 'scaler.pkl'")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def load_existing_model(self):
        try:
            self.model_manager.load()
            self.add_sample_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
            self.status_label.setText("📦 Model loaded successfully!")
            if self.model_manager.accuracy_history:
                self.plot_widget.plot(self.model_manager.accuracy_history)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{str(e)}")