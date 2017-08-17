# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CTCOGUI.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1340, 846)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.imageView = PlotWidget(self.centralwidget)
        self.imageView.setGeometry(QtCore.QRect(10, 10, 512, 512))
        self.imageView.setObjectName("imageView")
        self.timeScroll = QtWidgets.QScrollBar(self.centralwidget)
        self.timeScroll.setGeometry(QtCore.QRect(10, 530, 511, 16))
        self.timeScroll.setMaximum(100)
        self.timeScroll.setSliderPosition(0)
        self.timeScroll.setOrientation(QtCore.Qt.Horizontal)
        self.timeScroll.setObjectName("timeScroll")
        self.layerScroll = QtWidgets.QScrollBar(self.centralwidget)
        self.layerScroll.setGeometry(QtCore.QRect(530, 10, 16, 511))
        self.layerScroll.setOrientation(QtCore.Qt.Vertical)
        self.layerScroll.setObjectName("layerScroll")
        self.contrastSlider = QtWidgets.QSlider(self.centralwidget)
        self.contrastSlider.setGeometry(QtCore.QRect(10, 560, 511, 22))
        self.contrastSlider.setMinimum(1)
        self.contrastSlider.setMaximum(4000)
        self.contrastSlider.setProperty("value", 1330)
        self.contrastSlider.setSliderPosition(1330)
        self.contrastSlider.setOrientation(QtCore.Qt.Horizontal)
        self.contrastSlider.setObjectName("contrastSlider")
        self.brightnessSlider = QtWidgets.QSlider(self.centralwidget)
        self.brightnessSlider.setGeometry(QtCore.QRect(560, 10, 22, 511))
        self.brightnessSlider.setMinimum(-1000)
        self.brightnessSlider.setMaximum(2000)
        self.brightnessSlider.setProperty("value", 40)
        self.brightnessSlider.setSliderPosition(40)
        self.brightnessSlider.setOrientation(QtCore.Qt.Vertical)
        self.brightnessSlider.setObjectName("brightnessSlider")
        self.createROI_btn = QtWidgets.QPushButton(self.centralwidget)
        self.createROI_btn.setGeometry(QtCore.QRect(30, 590, 91, 32))
        self.createROI_btn.setObjectName("createROI_btn")
        self.clearROI_btn = QtWidgets.QPushButton(self.centralwidget)
        self.clearROI_btn.setGeometry(QtCore.QRect(30, 620, 91, 32))
        self.clearROI_btn.setObjectName("clearROI_btn")
        self.spinBoxROI = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBoxROI.setGeometry(QtCore.QRect(330, 675, 51, 24))
        self.spinBoxROI.setMinimum(1)
        self.spinBoxROI.setMaximum(50)
        self.spinBoxROI.setProperty("value", 5)
        self.spinBoxROI.setObjectName("spinBoxROI")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(270, 670, 61, 31))
        self.label_4.setObjectName("label_4")
        self.baseROI_btn = QtWidgets.QPushButton(self.centralwidget)
        self.baseROI_btn.setGeometry(QtCore.QRect(220, 590, 101, 32))
        self.baseROI_btn.setObjectName("baseROI_btn")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(40, 660, 171, 16))
        self.label_5.setObjectName("label_5")
        self.xCoordTxt = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.xCoordTxt.setGeometry(QtCore.QRect(40, 680, 71, 21))
        self.xCoordTxt.setObjectName("xCoordTxt")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 680, 16, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(140, 680, 16, 16))
        self.label_7.setObjectName("label_7")
        self.yCoordTxt = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.yCoordTxt.setGeometry(QtCore.QRect(160, 680, 71, 21))
        self.yCoordTxt.setObjectName("yCoordTxt")
        self.plotView = PlotWidget(self.centralwidget)
        self.plotView.setGeometry(QtCore.QRect(790, 10, 541, 351))
        self.plotView.setObjectName("plotView")
        self.label_peaktime = QtWidgets.QLabel(self.centralwidget)
        self.label_peaktime.setGeometry(QtCore.QRect(1090, 640, 81, 16))
        self.label_peaktime.setObjectName("label_peaktime")
        self.peakTime = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.peakTime.setGeometry(QtCore.QRect(1179, 640, 111, 24))
        self.peakTime.setMouseTracking(False)
        self.peakTime.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.peakTime.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.peakTime.setObjectName("peakTime")
        self.MTT = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.MTT.setGeometry(QtCore.QRect(1179, 680, 111, 24))
        self.MTT.setMouseTracking(False)
        self.MTT.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.MTT.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.MTT.setObjectName("MTT")
        self.label_sterror = QtWidgets.QLabel(self.centralwidget)
        self.label_sterror.setGeometry(QtCore.QRect(1120, 600, 51, 21))
        self.label_sterror.setObjectName("label_sterror")
        self.standardError = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.standardError.setGeometry(QtCore.QRect(1180, 600, 111, 24))
        self.standardError.setMouseTracking(False)
        self.standardError.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.standardError.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.standardError.setObjectName("standardError")
        self.label_MTT = QtWidgets.QLabel(self.centralwidget)
        self.label_MTT.setGeometry(QtCore.QRect(1040, 680, 131, 20))
        self.label_MTT.setObjectName("label_MTT")
        self.cardiacOutput = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.cardiacOutput.setGeometry(QtCore.QRect(1180, 560, 111, 24))
        self.cardiacOutput.setMouseTracking(False)
        self.cardiacOutput.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.cardiacOutput.setReadOnly(True)
        self.cardiacOutput.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.cardiacOutput.setObjectName("cardiacOutput")
        self.label_CO = QtWidgets.QLabel(self.centralwidget)
        self.label_CO.setGeometry(QtCore.QRect(1030, 560, 141, 20))
        self.label_CO.setObjectName("label_CO")
        self.CurveFitResults = QtWidgets.QGroupBox(self.centralwidget)
        self.CurveFitResults.setGeometry(QtCore.QRect(860, 380, 401, 151))
        self.CurveFitResults.setObjectName("CurveFitResults")
        self.label_rsquared = QtWidgets.QLabel(self.CurveFitResults)
        self.label_rsquared.setGeometry(QtCore.QRect(30, 80, 31, 16))
        self.label_rsquared.setObjectName("label_rsquared")
        self.alpha = QtWidgets.QPlainTextEdit(self.CurveFitResults)
        self.alpha.setGeometry(QtCore.QRect(20, 40, 101, 24))
        self.alpha.setMouseTracking(False)
        self.alpha.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.alpha.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.alpha.setObjectName("alpha")
        self.rsquared = QtWidgets.QPlainTextEdit(self.CurveFitResults)
        self.rsquared.setGeometry(QtCore.QRect(20, 100, 121, 24))
        self.rsquared.setMouseTracking(False)
        self.rsquared.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.rsquared.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.rsquared.setObjectName("rsquared")
        self.beta = QtWidgets.QPlainTextEdit(self.CurveFitResults)
        self.beta.setGeometry(QtCore.QRect(150, 40, 101, 24))
        self.beta.setMouseTracking(False)
        self.beta.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.beta.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.beta.setObjectName("beta")
        self.AUC = QtWidgets.QPlainTextEdit(self.CurveFitResults)
        self.AUC.setGeometry(QtCore.QRect(240, 100, 141, 24))
        self.AUC.setMouseTracking(False)
        self.AUC.setAcceptDrops(True)
        self.AUC.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.AUC.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.AUC.setObjectName("AUC")
        self.label_beta = QtWidgets.QLabel(self.CurveFitResults)
        self.label_beta.setGeometry(QtCore.QRect(160, 20, 31, 16))
        self.label_beta.setObjectName("label_beta")
        self.label_alpha = QtWidgets.QLabel(self.CurveFitResults)
        self.label_alpha.setGeometry(QtCore.QRect(30, 20, 41, 16))
        self.label_alpha.setObjectName("label_alpha")
        self.label_toffset = QtWidgets.QLabel(self.CurveFitResults)
        self.label_toffset.setGeometry(QtCore.QRect(290, 20, 31, 16))
        self.label_toffset.setObjectName("label_toffset")
        self.t0 = QtWidgets.QPlainTextEdit(self.CurveFitResults)
        self.t0.setGeometry(QtCore.QRect(280, 40, 101, 24))
        self.t0.setMouseTracking(False)
        self.t0.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.t0.setTextInteractionFlags(QtCore.Qt.TextSelectableByKeyboard|QtCore.Qt.TextSelectableByMouse)
        self.t0.setObjectName("t0")
        self.label_AUC = QtWidgets.QLabel(self.CurveFitResults)
        self.label_AUC.setGeometry(QtCore.QRect(250, 80, 31, 16))
        self.label_AUC.setObjectName("label_AUC")
        self.calcAndPlot_btn = QtWidgets.QPushButton(self.centralwidget)
        self.calcAndPlot_btn.setEnabled(True)
        self.calcAndPlot_btn.setGeometry(QtCore.QRect(420, 590, 111, 32))
        self.calcAndPlot_btn.setObjectName("calcAndPlot_btn")
        self.HUvalues = QtWidgets.QTableWidget(self.centralwidget)
        self.HUvalues.setGeometry(QtCore.QRect(640, 40, 101, 751))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.HUvalues.sizePolicy().hasHeightForWidth())
        self.HUvalues.setSizePolicy(sizePolicy)
        self.HUvalues.setMouseTracking(True)
        self.HUvalues.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.HUvalues.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.HUvalues.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.HUvalues.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.HUvalues.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.HUvalues.setGridStyle(QtCore.Qt.SolidLine)
        self.HUvalues.setRowCount(25)
        self.HUvalues.setColumnCount(2)
        self.HUvalues.setObjectName("HUvalues")
        self.HUvalues.horizontalHeader().setVisible(False)
        self.HUvalues.horizontalHeader().setCascadingSectionResizes(False)
        self.HUvalues.horizontalHeader().setDefaultSectionSize(50)
        self.HUvalues.verticalHeader().setVisible(False)
        self.HUvalues.verticalHeader().setCascadingSectionResizes(False)
        self.HUvalues.verticalHeader().setSortIndicatorShown(False)
        self.resetPlot = QtWidgets.QPushButton(self.centralwidget)
        self.resetPlot.setGeometry(QtCore.QRect(420, 620, 93, 32))
        self.resetPlot.setObjectName("resetPlot")
        self.HUtoIodineConversion = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.HUtoIodineConversion.setGeometry(QtCore.QRect(770, 600, 81, 24))
        self.HUtoIodineConversion.setMouseTracking(False)
        self.HUtoIodineConversion.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.HUtoIodineConversion.setObjectName("HUtoIodineConversion")
        self.label_HUConversion = QtWidgets.QLabel(self.centralwidget)
        self.label_HUConversion.setGeometry(QtCore.QRect(860, 600, 151, 16))
        self.label_HUConversion.setObjectName("label_HUConversion")
        self.label_baseline = QtWidgets.QLabel(self.centralwidget)
        self.label_baseline.setGeometry(QtCore.QRect(860, 560, 51, 20))
        self.label_baseline.setObjectName("label_baseline")
        self.timeInterval = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.timeInterval.setGeometry(QtCore.QRect(770, 640, 81, 24))
        self.timeInterval.setMouseTracking(False)
        self.timeInterval.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.timeInterval.setReadOnly(True)
        self.timeInterval.setObjectName("timeInterval")
        self.baselineInput = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.baselineInput.setGeometry(QtCore.QRect(770, 560, 81, 24))
        self.baselineInput.setMouseTracking(False)
        self.baselineInput.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.baselineInput.setReadOnly(True)
        self.baselineInput.setObjectName("baselineInput")
        self.label_tinterval = QtWidgets.QLabel(self.centralwidget)
        self.label_tinterval.setGeometry(QtCore.QRect(860, 640, 101, 20))
        self.label_tinterval.setObjectName("label_tinterval")
        self.createCSV = QtWidgets.QPushButton(self.centralwidget)
        self.createCSV.setEnabled(False)
        self.createCSV.setGeometry(QtCore.QRect(1172, 710, 121, 32))
        self.createCSV.setObjectName("createCSV")
        self.clearBASE_btn = QtWidgets.QPushButton(self.centralwidget)
        self.clearBASE_btn.setGeometry(QtCore.QRect(220, 620, 113, 32))
        self.clearBASE_btn.setObjectName("clearBASE_btn")
        self.MPAmean = QtWidgets.QTextBrowser(self.centralwidget)
        self.MPAmean.setGeometry(QtCore.QRect(130, 720, 101, 21))
        self.MPAmean.setObjectName("MPAmean")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 720, 101, 20))
        self.label.setObjectName("label")
        self.BASEmean = QtWidgets.QTextBrowser(self.centralwidget)
        self.BASEmean.setGeometry(QtCore.QRect(130, 750, 101, 21))
        self.BASEmean.setObjectName("BASEmean")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 750, 91, 20))
        self.label_2.setObjectName("label_2")
        self.DirSelect = QtWidgets.QPushButton(self.centralwidget)
        self.DirSelect.setGeometry(QtCore.QRect(420, 730, 131, 41))
        self.DirSelect.setObjectName("DirSelect")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(650, 20, 71, 20))
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1340, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Cardiac Output Calculator"))
        self.createROI_btn.setText(_translate("MainWindow", "MPA ROI\'s"))
        self.clearROI_btn.setText(_translate("MainWindow", "Clear MPA"))
        self.label_4.setText(_translate("MainWindow", "ROI Size:"))
        self.baseROI_btn.setText(_translate("MainWindow", "Baseline ROI"))
        self.label_5.setText(_translate("MainWindow", "Coordinates of ROI creation"))
        self.label_6.setText(_translate("MainWindow", "x:"))
        self.label_7.setText(_translate("MainWindow", "y:"))
        self.label_peaktime.setText(_translate("MainWindow", "Time to Peak (s)"))
        self.label_sterror.setText(_translate("MainWindow", "st. error"))
        self.label_MTT.setText(_translate("MainWindow", "Mean Transit Time (s)"))
        self.label_CO.setText(_translate("MainWindow", "Cardiac Output (L/min)"))
        self.CurveFitResults.setTitle(_translate("MainWindow", "Curve Fit Results"))
        self.label_rsquared.setText(_translate("MainWindow", "R^2"))
        self.label_beta.setText(_translate("MainWindow", "beta"))
        self.label_alpha.setText(_translate("MainWindow", "alpha"))
        self.label_toffset.setText(_translate("MainWindow", "t0 (s)"))
        self.label_AUC.setText(_translate("MainWindow", "AUC"))
        self.calcAndPlot_btn.setText(_translate("MainWindow", "Caclulate Data"))
        self.resetPlot.setText(_translate("MainWindow", "Reset Data"))
        self.label_HUConversion.setText(_translate("MainWindow", "HU/Iodine (HU/(mg/mL)) "))
        self.label_baseline.setText(_translate("MainWindow", "Baseline"))
        self.label_tinterval.setText(_translate("MainWindow", "Time Interval (s)"))
        self.createCSV.setText(_translate("MainWindow", "Create a CSV"))
        self.clearBASE_btn.setText(_translate("MainWindow", "Clear Baseline"))
        self.label.setText(_translate("MainWindow", "MPA ROI mean:"))
        self.label_2.setText(_translate("MainWindow", "Baseline mean:"))
        self.DirSelect.setText(_translate("MainWindow", "Select Directory"))
        self.label_3.setText(_translate("MainWindow", "t (s)       HU"))

from pyqtgraph import PlotWidget
