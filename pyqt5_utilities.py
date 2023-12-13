import json

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
    QPushButton,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QButtonGroup,
    QRadioButton,
    QWidget,
    QDialog,
    QLineEdit,
    QSpinBox,
    QTextEdit,
    QProgressBar,
    QHBoxLayout,
)
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
from dense_network import *


# def browse(self):
#     ui_file, _ = QFileDialog.getOpenFileName(self, "Select .ui File", "", "UI Files (*.ui)")
#     if ui_file:
#         self.ui_path_edit.setText(ui_file)

# works with .line_edit :QLineEdit
def browse(
    parent=None, caption="Select .csv file", directory="", filter="Csv Files (*.csv)"
):
    file_name, _ = QFileDialog.getOpenFileName(
        parent=parent, caption=caption, directory=directory, filter=filter
    )
    if file_name:
        #  assert file is .csv
        if not file_name.endswith(".csv"):
            print("file does not end with .csv")
            QMessageBox.critical(
                parent.centralwidget, "Invalid File", "Please select a .csv file."
            )
            return
        parent.line_edit.setText(file_name)


# works with .browse_label :QLabel
def browse_2(
    parent=None, caption="Select .csv file", directory="", filter="Csv Files (*.csv)"
):
    file_name, _ = QFileDialog.getOpenFileName(
        parent=parent, caption=caption, directory=directory, filter=filter
    )
    if file_name:
        #  assert file is .csv
        if not file_name.endswith(".csv"):
            print("file does not end with .csv")
            QMessageBox.critical(
                parent.centralwidget, "Invalid File", "Please select a .csv file."
            )
            return
        label_text = file_name[file_name.rindex("/") + 1 :]
        parent.browse_label.setText(label_text)
        parent.csv_data = pd.read_csv(file_name)
        forced_classification_popup(parent)
        feature_list, target = list_features(parent, parent.csv_data.columns.to_list())
        if len(feature_list) == 0 or target == -1:
            QMessageBox.critical(None, "Error", "Please select features and a target")
            return
        nn_shape = neural_net_shape_dialog(parent)
        if nn_shape is None:
            return
        n_epochs, ret = epoch_selector()
        if ret == 0:
            return
        parent.feature_indices = feature_list
        parent.target_index = target
        parent.hidden_shape = nn_shape
        parent.n_epochs = n_epochs
        save_model(parent)


def neural_net_shape_dialog(parent):
    def assertions(text_value: str):
        try:
            list(map(int, text_value.split("x")))
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
            error_msg.setWindowTitle("Invalid form")
            error_msg.setText("Please enter shape format as specified")
            error_msg.setStandardButtons(QMessageBox.Ok)
            error_msg.exec_()

    dialog = QDialog()
    dialog.setWindowTitle("Neural Net layer shape")
    layout = QVBoxLayout()

    label = QLabel(
        "In the text below specify the shape of the dense nn\n"
        "for example for a 16x16x32 write '16x16x32' without apostrophe\n"
        "only integers are accepted"
    )
    layout.addWidget(label)

    shape_text = QLineEdit()
    layout.addWidget(shape_text)
    text = None

    ok_button = QPushButton("OK", dialog)
    ok_button.clicked.connect(on_button_clicked)
    layout.addWidget(ok_button)

    dialog.setLayout(layout)
    dialog.exec_()

    return list(map(int, text.split("x")))


def epoch_selector():

    value = 1

    def on_value_change():
        nonlocal spin_box
        nonlocal label
        nonlocal value
        value = spin_box.value()

    dialog = QDialog()
    dialog.setWindowTitle("Epochs")

    layout = QVBoxLayout()
    label = QLabel("Please specify number of epochs\nin the training", dialog)

    spin_box = QSpinBox(dialog)
    spin_box.setRange(1, 10_000)
    spin_box.valueChanged.connect(on_value_change)

    ok_button = QPushButton("OK", dialog)
    # ok_button.clicked.connect(ok_confirmation)
    ok_button.clicked.connect(dialog.accept)

    layout.addWidget(label)
    layout.addWidget(spin_box)
    layout.addWidget(ok_button)
    dialog.setLayout(layout)
    ret = dialog.exec_()

    return value, ret


def save_model(parent):
    def training():
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, ext = QFileDialog.getSaveFileName(
            parent,
            "Save Model As",
            "",
            "Keras Models (*.h5);;All Files (*)",
            options=options,
        )
        if fileName:
            parent.save_location = fileName
            # required for linux user
            if not parent.save_location.endswith(".h5"):
                filename, ext = os.path.splitext(parent.save_location)
                parent.save_location = filename + ".h5"
            model = run_nn(parent)
            if model:
                model.save(parent.save_location)
                dialog.accept()

    dialog = QDialog()
    dialog.setWindowTitle("Save and train")

    layout = QVBoxLayout()
    label = QLabel("Choose where to save the model", dialog)
    save_button = QPushButton("Browse", dialog)
    save_button.clicked.connect(training)

    layout.addWidget(label)
    layout.addWidget(save_button)
    dialog.setLayout(layout)

    dialog.exec_()


def run_nn(parent):
    def final_message(loss_value):
        msg = QMessageBox()
        msg.setWindowTitle("Model saved")
        msg.setText(
            "Model saved in %s with loss %.4f" % (parent.save_location, loss_value)
        )
        msg.setStandardButtons(QMessageBox.Ok)  # Add OK button
        msg.exec_()

    X = parent.csv_data.iloc[:, parent.feature_indices].to_numpy()
    y = parent.csv_data.iloc[:, parent.target_index].to_numpy()
    if parent.classification:
        y = LabelEncoder().fit_transform(y)
        y = one_hot(y)
        model = build_nn_classifier(
            nn_shape=np.array(parent.hidden_shape),
            in_shape=X.shape[1],
            out_shape=y.shape[1],
        )
        # model.fit(X, y, epochs=parent.n_epochs)
        model, loss = train_and_progress_bar(model, X, y, parent.n_epochs)
        final_message(loss)
        return model
    else:
        y = LabelEncoder().fit_transform(y)
        y = one_hot(y)
        model = build_nn_classifier(
            nn_shape=np.array(parent.hidden_shape),
            in_shape=X.shape[1],
            out_shape=y.shape[1],
        )
        try:
            # model.fit(X, y, epochs=parent.n_epochs)
            model, loss = train_and_progress_bar(model, X, y, parent.n_epochs)
            final_message(loss)
            return model
        except ValueError:
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setWindowTitle("Error")
            error_msg.setText("Target data is not in numerical form")
            error_msg.setStandardButtons(QMessageBox.Ok)
            error_msg.exec_()
            return None


def train_and_progress_bar(model, X, y, n_epochs):
    class TrainingThread(QThread):
        epoch_begin = pyqtSignal(int)  # Signal to notify the start of an epoch
        training_completed = pyqtSignal()

        def __init__(self):
            super().__init__()
            self.loss = None

        def run(self):
            self.loss = self.train_model()
            self.training_completed.emit()

        def train_model(self):
            progress_callback = TrainingProgressCallback(
                lambda e: self.epoch_begin.emit(e)
            )
            history = model.fit(X, y, epochs=n_epochs, callbacks=[progress_callback])
            return history.history["loss"][-1]

    def on_epoch_begin(epoch):
        nonlocal progress_bar
        progress_bar.setValue(epoch)

    def on_training_complete():
        dialog.accept()

    dialog = QDialog()
    dialog.setWindowTitle("Training")
    layout = QVBoxLayout()
    progress_bar = QProgressBar(dialog)
    progress_bar.setMaximum(n_epochs)
    layout.addWidget(progress_bar)
    dialog.setLayout(layout)

    training_thread = TrainingThread()
    training_thread.epoch_begin.connect(on_epoch_begin)
    training_thread.start()
    training_thread.training_completed.connect(on_training_complete)

    dialog.exec_()

    return model, training_thread.loss


def vanilla_popup():
    msg_box = QMessageBox(
        QMessageBox.Information, "Information", "pop-up message", QMessageBox.Ok
    )
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
        for index in range(feature_checklist.count()):
            item = feature_checklist.item(index)
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

    dialog = QDialog()
    dialog.setWindowTitle("Feature selection")

    feature_label = QLabel("Select features to use in the model:")
    feature_checklist = QListWidget()
    select_all = QPushButton("Select All")
    select_all.clicked.connect(select_all_features)

    target_label = QLabel("Select target:")
    radio_button_list = QButtonGroup()

    for f in feature_list:
        item = QListWidgetItem(f)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # Make item checkable
        item.setCheckState(Qt.Unchecked)  # Initial state is unchecked
        feature_checklist.addItem(item)

    r_b_s = []  # keep reference to prevent garbage collection
    for f in feature_list:
        radio_button = QRadioButton(f)
        radio_button.setStyleSheet(
            "QRadioButton { margin: 1px; padding: 1px; spacing: 1px; }"
        )
        radio_button_list.addButton(
            radio_button
        )  # Add button to group to ensure single selection
        r_b_s.append(radio_button)  # Keep the reference

    target_layout = QVBoxLayout()
    target_button_group = QButtonGroup()

    for f in feature_list:
        radio_button = QRadioButton(f)
        radio_button.setStyleSheet(
            "QRadioButton { margin: 1px; padding: 1px; spacing: 1px; }"
        )
        target_button_group.addButton(
            radio_button
        )  # Add button to group to ensure single selection
        # target_layout.addWidget(radio_button)
    target_widget = QWidget()
    target_widget.setLayout(target_layout)

    left_layout = QVBoxLayout()
    left_layout.addWidget(feature_label)
    left_layout.addWidget(feature_checklist)
    left_layout.addWidget(select_all)

    right_layout = QVBoxLayout()
    right_layout.addWidget(target_label)
    for f in feature_list:
        radio_button = QRadioButton(f)
        radio_button.setStyleSheet(
            "QRadioButton { margin: 1px; padding: 1px; spacing: 1px; }"
        )
        target_button_group.addButton(
            radio_button
        )  # Add button to group to ensure single selection
        right_layout.addWidget(radio_button)
    right_layout.addStretch(1)

    ok_button = QPushButton("OK")
    ok_button.clicked.connect(dialog.accept)

    main_layout = QHBoxLayout()
    main_layout.addLayout(left_layout)
    main_layout.addLayout(right_layout)
    main_layout.addWidget(ok_button)
    dialog.setLayout(main_layout)

    dialog.exec_()

    f_selected_indices, _ = get_selected_features(feature_checklist)
    t_selected_index = np.abs(target_button_group.checkedId()) - 2

    return f_selected_indices, t_selected_index


def legacy_list_features(parent, feature_list):
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
        radio_button.setStyleSheet(
            "QRadioButton { margin: 1px; padding: 1px; spacing: 1px; }"
        )
        target_button_group.addButton(
            radio_button
        )  # Add button to group to ensure single selection
        target_layout.addWidget(radio_button)
    target_widget = QWidget()
    target_widget.setLayout(target_layout)

    feature_box.select_all_features = QPushButton("Select All")
    feature_box.select_all_features.clicked.connect(select_all_features)

    layout = feature_box.layout()
    layout.addWidget(feature_box.feature_checklist, 1, 0)
    layout.addWidget(feature_label, 0, 0)
    layout.addWidget(feature_box.select_all_features, 2, 0)
    # layout.addWidget(feature_box.target_checklist, 0, 1)
    layout.addWidget(target_widget, 1, 1)
    layout.addWidget(target_label, 0, 1)

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
