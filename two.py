# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'two.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from pyqt5_utilities import browse_2


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")
        self.csv_data = None
        self.classification = True
        self.hidden_shape = None
        self.n_epochs = None
        self.save_location = None
        self.feature_indices, self.target_index = None, None
        self.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.browse_button = QtWidgets.QPushButton(self.centralwidget)
        self.browse_button.setGeometry(QtCore.QRect(680, 30, 95, 27))
        self.browse_button.setObjectName("browse_button")
        self.browse_label = QtWidgets.QLabel(self.centralwidget)
        self.browse_label.setGeometry(QtCore.QRect(380, 30, 291, 20))
        self.browse_label.setText("")
        self.browse_label.setObjectName("browse_label")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

        self.browse_button.clicked.connect(lambda: browse_2(self))

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "Neural Net Elaborator"))
        self.browse_button.setText(_translate("MainWindow", "Browse"))


# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     window = Ui_MainWindow()
#     window.show()
#     sys.exit(app.exec_())

def run_app():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())

    # data = np.arange(10)
    # with open(save_path, 'w') as file:
    #     json.dump(data.tolist(), file)

