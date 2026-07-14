"""
Dark mode minimalist stylesheet for Regime Discovery UI.
Single vibrant blue accent: #00b4ff
"""

DARK_STYLESHEET = """
/* Global */
QWidget {
    background-color: #0f0f0f;
    color: #e0e0e0;
    font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #0f0f0f;
}

/* Sidebar / Panels */
QFrame#sidebar, QFrame#configPanel, QFrame#repoPanel {
    background-color: #1a1a1a;
    border-right: 1px solid #2a2a2a;
}

QFrame#card {
    background-color: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
}

/* Labels */
QLabel {
    color: #e0e0e0;
    background: transparent;
}

QLabel#title {
    font-size: 18px;
    font-weight: 600;
    color: #ffffff;
}

QLabel#sectionTitle {
    font-size: 14px;
    font-weight: 600;
    color: #00b4ff;
    padding-top: 8px;
    padding-bottom: 4px;
}

QLabel#subtitle {
    color: #a0a0a0;
    font-size: 12px;
}

QLabel#valueLabel {
    color: #00b4ff;
    font-weight: 500;
}

/* Buttons */
QPushButton {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2a2a2a;
    border-color: #00b4ff;
}

QPushButton:pressed {
    background-color: #003d5c;
}

QPushButton#primaryButton {
    background-color: #00b4ff;
    color: #0f0f0f;
    border: none;
    font-weight: 600;
}

QPushButton#primaryButton:hover {
    background-color: #33c3ff;
}

QPushButton#primaryButton:pressed {
    background-color: #0090cc;
}

QPushButton#primaryButton:disabled {
    background-color: #3a3a3a;
    color: #707070;
}

/* Inputs */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 10px;
    selection-background-color: #00b4ff;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #00b4ff;
}

QComboBox {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 10px;
}

QComboBox:hover, QComboBox:focus {
    border-color: #00b4ff;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #1e1e1e;
    color: #e0e0e0;
    selection-background-color: #00b4ff;
    selection-color: #0f0f0f;
    border: 1px solid #3a3a3a;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #3a3a3a;
    height: 6px;
    background: #2a2a2a;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #00b4ff;
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #33c3ff;
}

QSlider::sub-page:horizontal {
    background: #00b4ff;
    border-radius: 3px;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #2a2a2a;
    background-color: #0f0f0f;
    border-radius: 0 0 6px 6px;
}

QTabBar::tab {
    background-color: #1a1a1a;
    color: #a0a0a0;
    border: 1px solid #2a2a2a;
    border-bottom: none;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #0f0f0f;
    color: #00b4ff;
    border-bottom: 2px solid #00b4ff;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background-color: #252525;
    color: #e0e0e0;
}

/* Table */
QTableWidget, QTableView {
    background-color: #141414;
    alternate-background-color: #1a1a1a;
    color: #e0e0e0;
    gridline-color: #2a2a2a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    selection-background-color: #003d5c;
    selection-color: #ffffff;
}

QHeaderView::section {
    background-color: #1e1e1e;
    color: #00b4ff;
    padding: 8px;
    border: none;
    border-right: 1px solid #2a2a2a;
    border-bottom: 1px solid #2a2a2a;
    font-weight: 600;
}

QTableWidget::item {
    padding: 6px;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #1a1a1a;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #3a3a3a;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #00b4ff;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #1a1a1a;
    height: 10px;
}

QScrollBar::handle:horizontal {
    background: #3a3a3a;
    border-radius: 5px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: #00b4ff;
}

/* Checkboxes / Radio */
QCheckBox, QRadioButton {
    color: #e0e0e0;
    spacing: 8px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    background: #1e1e1e;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #00b4ff;
    border-color: #00b4ff;
}

/* GroupBox */
QGroupBox {
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: 600;
    color: #00b4ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

/* Splitter */
QSplitter::handle {
    background-color: #2a2a2a;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

/* Status / Progress */
QStatusBar {
    background-color: #1a1a1a;
    color: #a0a0a0;
    border-top: 1px solid #2a2a2a;
}

QProgressBar {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    background: #1e1e1e;
    text-align: center;
    color: #e0e0e0;
}

QProgressBar::chunk {
    background-color: #00b4ff;
    border-radius: 3px;
}

/* Tooltips */
QToolTip {
    background-color: #1e1e1e;
    color: #e0e0e0;
    border: 1px solid #00b4ff;
    border-radius: 4px;
    padding: 4px;
}
"""
