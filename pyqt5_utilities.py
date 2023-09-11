import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import os


# def browse(self):
#     ui_file, _ = QFileDialog.getOpenFileName(self, "Select .ui File", "", "UI Files (*.ui)")
#     if ui_file:
#         self.ui_path_edit.setText(ui_file)

def browse(parent=None, caption='Select .csv file', directory='', filter='Csv Files (*.csv)'):
    file_name, _ = QFileDialog.getOpenFileName(
        parent=parent, caption=caption, directory=directory, filter=filter)
    if file_name:
        #  assert file is .csv
        if not file_name.endswith('.csv'):
            print('file does not end with .csv')
            QMessageBox.critical(parent.centralwidget, 'Invalid File', 'Please select a .csv file.')
            return
        parent.line_edit.setText(file_name)


def convert(self):
    ui_file = self.ui_path_edit.text().strip()
    if ui_file:
        output_file = os.path.splitext(ui_file)[0] + ".py"
        os.system(f"pyuic5 -x '{ui_file}' -o '{output_file}'")
        self.status_label.setText(f"Status: Converted to {output_file}")
    else:
        self.status_label.setText("Status: No file selected!")

























































































