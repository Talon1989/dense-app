import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QLabel, QPushButton,
                             QMessageBox, QListWidget, QListWidgetItem, QVBoxLayout, QButtonGroup, QRadioButton,
                             QWidget, QDialog, QLineEdit)
import os
import pandas as pd


# def browse(self):
#     ui_file, _ = QFileDialog.getOpenFileName(self, "Select .ui File", "", "UI Files (*.ui)")
#     if ui_file:
#         self.ui_path_edit.setText(ui_file)

# works with .line_edit :QLineEdit
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


# works with .browse_label :QLabel
def browse_2(parent=None, caption='Select .csv file', directory='', filter='Csv Files (*.csv)'):
    file_name, _ = QFileDialog.getOpenFileName(
        parent=parent, caption=caption, directory=directory, filter=filter)
    if file_name:
        #  assert file is .csv
        if not file_name.endswith('.csv'):
            print('file does not end with .csv')
            QMessageBox.critical(parent.centralwidget, 'Invalid File', 'Please select a .csv file.')
            return
        label_text = file_name[file_name.rindex('/')+1:]
        parent.browse_label.setText(label_text)
        csv_file = pd.read_csv(file_name)
        # print('Features present in the .csv file are\n%s' % csv_file.columns.to_list())
        forced_classification_popup(parent)
        feature_list, target = list_features(parent, csv_file.columns.to_list())
        if len(feature_list) == 0 or target == -1:
            QMessageBox.critical(None, 'Error', 'Please select features and a target')
            return
        nn_shape = neural_net_shape_dialog(parent)
        print(nn_shape)


def neural_net_shape_dialog(parent):

    def assertions(text_value: str):
        try:
            list(map(int, text_value.split('x')))
            return True
        except ValueError:
            return False

    def on_button_clicked():
        nonlocal text
        text = shape_text.text()
        if assertions(text):
            dialog.accept()
        else:
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setWindowTitle('Invalid form')
            error_msg.setText("Please enter shape format as specified")
            error_msg.setStandardButtons(QMessageBox.Ok)
            error_msg.exec_()


    dialog = QDialog()
    dialog.setWindowTitle('Neural Net layer shape')
    layout = QVBoxLayout()

    label = QLabel("In the text below specify the shape of the dense nn\n"
                   "for example for a 16x16x32 write '16x16x32' without apostrophe\n"
                   "only integers are accepted")
    layout.addWidget(label)

    shape_text = QLineEdit()
    layout.addWidget(shape_text)
    text = None

    ok_button = QPushButton('OK', dialog)
    ok_button.clicked.connect(on_button_clicked)
    layout.addWidget(ok_button)

    dialog.setLayout(layout)
    dialog.exec_()

    return text






def vanilla_popup():
    msg_box = QMessageBox(QMessageBox.Information, 'Information', 'pop-up message', QMessageBox.Ok)
    msg_box.exec_()


def forced_classification_popup(parent=None):
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle("Type of learning")
    msg_box.setText("Choose a type of learning")
    classification = msg_box.addButton("Classification", QMessageBox.ActionRole)
    regression = msg_box.addButton("Regression", QMessageBox.ActionRole)
    msg_box.exec_()
    if msg_box.clickedButton() == classification:
        parent.classification = True
    elif msg_box.clickedButton() == regression:
        parent.classification = False


def list_features(parent, feature_list):
    def select_all_features():
        for index in range(feature_box.feature_checklist.count()):
            item = feature_box.feature_checklist.item(index)
            item.setCheckState(Qt.Checked)

    def get_selected_features(checklist: QListWidget):
        selected_indices, non_selected_indices = [], []
        for idx in range(checklist.count()):
            item = checklist.item(idx)
            if item.checkState() == Qt.Checked:
                selected_indices.append(idx)
            else:
                non_selected_indices.append(idx)
        return selected_indices, non_selected_indices

    from PyQt5.QtCore import Qt
    feature_box = QMessageBox()
    feature_box.setStyleSheet("QLabel{min-width: 400px; min-height: 300px}")
    feature_box.setWindowTitle("Feature selection")
    feature_label = QLabel("Select features to use in the model:")
    target_label = QLabel("Select target:")

    feature_box.feature_checklist = QListWidget()
    feature_box.feature_checklist.setMinimumWidth(250)

    for f in feature_list:
        item = QListWidgetItem(f)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # Make item checkable
        item.setCheckState(Qt.Unchecked)  # Initial state is unchecked
        feature_box.feature_checklist.addItem(item)

    # feature_box.target_checklist = QListWidget()
    # feature_box.target_checklist.setMinimumWidth(250)
    target_layout = QVBoxLayout()
    target_button_group = QButtonGroup(feature_box)

    # for f in feature_list:
    #     item = QListWidgetItem(f)
    #     item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # Make item checkable
    #     item.setCheckState(Qt.Unchecked)  # Initial state is unchecked
    #     feature_box.target_checklist.addItem(item)
    for f in feature_list:
        radio_button = QRadioButton(f)
        radio_button.setStyleSheet("QRadioButton { margin: 1px; padding: 1px; spacing: 1px; }")
        target_button_group.addButton(radio_button)  # Add button to group to ensure single selection
        target_layout.addWidget(radio_button)
    target_widget = QWidget()
    target_widget.setLayout(target_layout)

    feature_box.select_all_features = QPushButton('Select All')
    feature_box.select_all_features.clicked.connect(select_all_features)

    layout = feature_box.layout()
    layout.addWidget(feature_box.feature_checklist, 0, 0)
    layout.addWidget(feature_label, 1, 0)
    layout.addWidget(feature_box.select_all_features, 2, 0)
    # layout.addWidget(feature_box.target_checklist, 0, 1)
    layout.addWidget(target_widget, 0, 1)
    layout.addWidget(target_label, 1, 1)

    feature_box.exec_()

    f_selected_indices, _ = get_selected_features(feature_box.feature_checklist)
    t_selected_index = np.abs(target_button_group.checkedId()) - 2

    return f_selected_indices, t_selected_index


def convert(self):
    ui_file = self.ui_path_edit.text().strip()
    if ui_file:
        output_file = os.path.splitext(ui_file)[0] + ".py"
        os.system(f"pyuic5 -x '{ui_file}' -o '{output_file}'")
        self.status_label.setText(f"Status: Converted to {output_file}")
    else:
        self.status_label.setText("Status: No file selected!")

























































































