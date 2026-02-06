import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import AdaptiveLearningWindow
from model_manager import ModelManager

def main():
    app = QApplication(sys.argv)
    model_mgr = ModelManager()
    window = AdaptiveLearningWindow(model_mgr)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()