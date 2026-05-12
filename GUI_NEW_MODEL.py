import sys

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QTextEdit,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView
)

from PyQt5.QtCore import Qt

import predict_all_models_bundle.predict_all_models as _pm


# Load models once at startup
_crf_model, _ = _pm.load_pickle_model(_pm.DEFAULT_CRF_MODEL_PATH)
_svm_model, _ = _pm.load_svm_model(_pm.DEFAULT_SVM_MODEL_PATH)


class UIOutline(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("PII Tool")
        self.setGeometry(100, 100, 1000, 500)

        # MAIN LAYOUTS
        main_layout = QVBoxLayout()
        content_layout = QHBoxLayout()

        # LEFT SIDE
        left_layout = QVBoxLayout()

        left_label = QLabel("Input Text")
        left_label.setAlignment(Qt.AlignCenter)
        left_label.setStyleSheet(
            "font-weight: bold; font-size: 12px;"
        )

        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText(
            "Input text here..."
        )

        left_layout.addWidget(left_label)
        left_layout.addWidget(self.input_box)

        # CENTER ARROW
        arrow = QLabel("→")
        arrow.setAlignment(Qt.AlignCenter)
        arrow.setStyleSheet("font-size: 40px;")

        # RIGHT SIDE
        right_layout = QVBoxLayout()

        # PREDICTION LABEL
        prediction_label = QLabel("Predicted Labels")
        prediction_label.setAlignment(Qt.AlignCenter)
        prediction_label.setStyleSheet(
            "font-weight: bold;"
        )

        # RESULT TABLE
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(
            ["Label", "Value"]
        )

        self.result_table.horizontalHeader().setStretchLastSection(True)

        self.result_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.Stretch
        )

        self.result_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.Stretch
        )

        # CONFIDENCE LABEL
        confidence_label = QLabel("Model Confidence")
        confidence_label.setAlignment(Qt.AlignCenter)
        confidence_label.setStyleSheet(
            "font-weight: bold;"
        )

        # SCORE TABLE
        self.score_table = QTableWidget()
        self.score_table.setColumnCount(2)
        self.score_table.setRowCount(5)

        self.score_table.setHorizontalHeaderLabels(
            ["Metric", "Score"]
        )

        self.score_table.horizontalHeader().setStretchLastSection(True)

        self.score_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.Stretch
        )

        self.score_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.Stretch
        )

        # ADD TO RIGHT LAYOUT
        right_layout.addWidget(prediction_label)
        right_layout.addWidget(self.result_table)

        right_layout.addWidget(confidence_label)
        right_layout.addWidget(self.score_table)

        # COMBINE
        content_layout.addLayout(left_layout, 3)
        content_layout.addWidget(arrow, 1)
        content_layout.addLayout(right_layout, 3)

        # BUTTON
        self.button = QPushButton("Process")
        self.button.setFixedWidth(150)

        self.button.clicked.connect(
            self.process_text
        )

        button_layout = QHBoxLayout()

        button_layout.addStretch()
        button_layout.addWidget(self.button)
        button_layout.addStretch()

        # FINAL ASSEMBLY
        main_layout.addLayout(content_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def process_text(self):

        text = self.input_box.toPlainText()

        if not text.strip():
            return

        # RUN PIPELINE
        normalized    = _pm.normalize_crf_text(text)
        regex_result  = _pm.regex_inference(normalized)
        crf_result    = _pm.crf_inference(_crf_model, normalized)
        candidates    = _pm.build_candidates(regex_result, crf_result)
        svm_result    = _pm.svm_inference(_svm_model, candidates)

        output = _pm.build_output(
            raw_text=text,
            normalized_text=normalized,
            regex_result=regex_result,
            crf_result=crf_result,
            svm_result=svm_result,
            candidates=candidates,
            crf_warnings=[],
            svm_warnings=[],
            crf_model_path=_pm.DEFAULT_CRF_MODEL_PATH,
            svm_model_path=_pm.DEFAULT_SVM_MODEL_PATH,
        )

        summary  = output["summary"]
        combined = summary["combined"]

        r     = summary["models"]["regex"]["pii_prob"]
        c     = summary["models"]["crf"]["pii_prob"]
        s     = summary["models"]["svm"]["pii_prob"]
        final = combined["pii_prob"]

        matches = regex_result["spans"]

        # CLEAR TABLES
        self.result_table.clearContents()
        self.score_table.clearContents()

        # PREDICTION TABLE
        if matches:

            self.result_table.setRowCount(len(matches))

            for row, item in enumerate(matches):

                self.result_table.setItem(
                    row,
                    0,
                    QTableWidgetItem(str(item.get("label", "Unknown")))
                )

                self.result_table.setItem(
                    row,
                    1,
                    QTableWidgetItem(str(item.get("value", "")))
                )

        else:

            self.result_table.setRowCount(1)

            self.result_table.setItem(
                0,
                0,
                QTableWidgetItem("No matches")
            )

            self.result_table.setItem(
                0,
                1,
                QTableWidgetItem("-")
            )

        # SCORE TABLE
        metrics = [
            ("Regex Score", f"{r:.3f}"),
            ("CRF Score",   f"{c:.3f}"),
            ("SVM Score",   f"{s:.3f}"),
            ("Final Score", f"{final:.3f}"),
            (
                "Verdict",
                "PII detected"
                if final > 0.5
                else "No PII detected"
            )
        ]

        self.score_table.setRowCount(
            len(metrics)
        )

        for row, (metric, value) in enumerate(metrics):

            self.score_table.setItem(
                row,
                0,
                QTableWidgetItem(metric)
            )

            self.score_table.setItem(
                row,
                1,
                QTableWidgetItem(value)
            )


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = UIOutline()
    window.show()

    sys.exit(app.exec_())