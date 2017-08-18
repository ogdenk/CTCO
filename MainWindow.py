from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
import pyqtgraph as pqg
from pyqtgraph import *
import ui_CTCO
import dicom
from subprocess import call
from PyQt5.QtWidgets import *
import pyqtgraph as pqg
from pyqtgraph import *
import PatientData
from scipy.optimize import curve_fit
import numpy as np
import sys
import csv
import datetime

class CTCOMain(QMainWindow, ui_CTCO.Ui_MainWindow):

    def __init__(self, parent=None):
        super(CTCOMain, self).__init__(parent)
        pqg.setConfigOption('background', '#f0f0f0')
        pqg.setConfigOption('foreground', '#2d3142')
        pqg.mkPen(color=(0, 97, 255))
        self.setupUi(self)
        self.HUtoIodineConversion.setPlainText("24")
        self.legend = False
        self.patient = PatientData.Patient()  # object that holds the data and does the calculation
        self.legendExists = False
        self.setMouseTracking(True)

        self.imv = pqg.ImageView(parent=self.imageView)
        self.lstFilesDCM = []  # create an empty list
        self.seenPos = []  # List to hold possible positions
        self.seenTime = []  # List to hold possible times

        #ROI creation
        #Coordinate Boxes
        self.xCoordTxt.setPlainText("0")
        self.yCoordTxt.setPlainText("0")
        #Changes ROI based on GUI inputs
        def resizeROI():
            self.roi.setSize([self.spinBoxROI.value(), self.spinBoxROI.value()])

        self.spinBoxROI.valueChanged.connect(resizeROI)

        self.roi = pqg.RectROI([0,0], [self.spinBoxROI.value(), self.spinBoxROI.value()])
        self.roiList = []
        self.ROIexists = False

        #ROI buttons
        #Creates the ROI by linking the roi to the image
        def createROI():
            QApplication.setOverrideCursor(QCursor(QtCore.Qt.CrossCursor))
            self.createROI_btn.setDown(True)
            self.roi.sigRegionChanged.connect(update)

        self.createROI_btn.clicked.connect(createROI)

        #Hides the ROI by removing the link to the image and moving it out of frame
        def clearROI():
            if self.ROIexists:
                self.roi.setParentItem(None)
                self.roi.setPos(-10000, 0)
                self.roi.setSize([self.spinBoxROI.value(), self.spinBoxROI.value()])
                self.roiList.clear()
                self.ROIexists = False
        def clearBASE():
            if self.BASEexists:
                self.BASEroi.setParentItem(None)
                self.BASEroi.setPos(-10000, 0)
                self.BASEroi.setSize([self.spinBoxROI.value(), self.spinBoxROI.value()])
                self.BASEroiSAVE.clear()
                self.BASEexists = False

        self.clearROI_btn.clicked.connect(clearROI)
        self.clearBASE_btn.clicked.connect(clearBASE)

        def moveBASE():
            if(self.BASEexists and self.timeScroll.sliderPosition()==0):
                self.BASEmean.setPlainText(str(self.BASEroi.getArrayRegion(self.finalArray[self.layerScroll.sliderPosition()][:, :, 0].T, self.imv.getImageItem()).mean()))


        #Changes data based on moving of ROI as it happens
        def update(roi):
            ROIarray = roi.getArrayRegion(self.finalArray[self.layerScroll.sliderPosition()][:, :, self.timeScroll.sliderPosition()].T, self.imv.getImageItem())
            self.MPAmean.setPlainText(
                str(self.roi.getArrayRegion(self.finalArray[self.layerScroll.sliderPosition()][:, :,
                                            self.timeScroll.sliderPosition()].T, self.imv.getImageItem()).mean()))
            np.fliplr(ROIarray)

        #Finds the mean of the data within each of the ROIs
        def collectMeans():
            self.readyToPlot=False
            mean = 0
            meanlst = []
            self.timeInterval.setPlainText(str(self.timeIntervalfloat))
            if self.BASEexists:
                tempBASE = self.BASEroi.saveState()
                if not self.ROIexists:
                    msgBox = QMessageBox()
                    msgBox.setText("There was an error:")
                    msgBox.setInformativeText("Ensure that the ROI boxes are set.")
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.setDefaultButton(QMessageBox.Ok)
                    msgBox.exec_()
            if self.ROIexists:
                if not self.BASEexists:
                    msgBox = QMessageBox()
                    msgBox.setText("There was an error:")
                    msgBox.setInformativeText("Ensure that the baseline ROI box is set.")
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.setDefaultButton(QMessageBox.Ok)
                    msgBox.exec_()
                update(self.roi)
                preMove()
                tempROI = self.roi
                tempState = tempROI.saveState()
                tempPlace = self.timeScroll.sliderPosition()
                for i in np.arange(0, self.nTime, 1):
                    self.imv.setImage(self.finalArray[self.layerScroll.sliderPosition()][:, :, i].T, autoRange=False, autoLevels=False)
                    self.roi.setState(self.roiList[i])
                    if self.BASEexists and i == 0:
                        self.BASEroi.setState(self.BASEroiSAVE)
                        BaseLineNum = round((self.BASEroi.getArrayRegion(self.finalArray[self.layerScroll.sliderPosition()][:, :, 0].T, self.imv.getImageItem()).mean()),4)#baseline
                        self.baselineInput.setPlainText(str(BaseLineNum))
                    else:
                        self.BASEroi.setPos(-100000,100000)
                    mean += self.roi.getArrayRegion(self.finalArray[self.layerScroll.sliderPosition()][:, :, i].T, self.imv.getImageItem()).mean()
                    meanlst.append(int(self.roi.getArrayRegion(self.finalArray[self.layerScroll.sliderPosition()][:, :, i].T, self.imv.getImageItem()).mean()))
                print(meanlst)#tablevalues
                mean = mean/self.nTime
                if self.BASEexists:
                    self.BASEroi.setState(tempBASE)
                    self.readyToPlot = True
                self.roi.setState(tempState)
                self.imv.setImage(self.finalArray[self.layerScroll.sliderPosition()][:, :, tempPlace].T, autoRange=False,autoLevels=False)
                #nonNumpyMeanlst = meanlst.tolist()#meanlst.astype(type('float', (float,), {}))
                for i in range(0,meanlst.__len__()):
                    self.HUvalues.setItem(i,1,QTableWidgetItem(str(meanlst[i])))
                    self.HUvalues.setItem(i,0,QTableWidgetItem(str(float(self.timeInterval.toPlainText())*i)))
            if self.readyToPlot:
                self.ApplyChecker()



        self.calcAndPlot_btn.clicked.connect(collectMeans)

        #Creation of Baseline ROI
        self.BASEroi = pqg.RectROI([0,0], [self.spinBoxROI.value(), self.spinBoxROI.value()])
        self.BASEroiSAVE = self.BASEroi.saveState()
        self.BASEexists = False

        def setBaseLine():
            QApplication.setOverrideCursor(QCursor(QtCore.Qt.CrossCursor))
            self.baseROI_btn.setDown(True)
            self.BASEroi.sigRegionChanged.connect(moveBASE)

        self.baseROI_btn.clicked.connect(setBaseLine)


        #Slider for time
        #self.timeScroll.setMaximum(self.nTime-1)
        def updateT():
            self.imv.setImage(self.finalArray[self.layerScroll.sliderPosition()][:, :, self.timeScroll.sliderPosition()].T, autoRange=False, autoLevels=False)
            if self.ROIexists:
                self.roi.setState(self.roiList[self.timeScroll.sliderPosition()])
                update(self.roi)
            if self.BASEexists:
                if self.timeScroll.sliderPosition()==0:
                    self.BASEroi.setState(self.BASEroiSAVE)
                else:
                    self.BASEroi.setPos(-1000000,0)

        #Saves current ROI state when the time is about to be changed
        def preMove():
            if self.ROIexists:
                self.roiList[self.timeScroll.sliderPosition()] = self.roi.saveState()
            if self.BASEexists:
                if self.timeScroll.sliderPosition() == 0:
                    self.BASEroiSAVE = self.BASEroi.saveState()

        self.timeScroll.sliderMoved.connect(updateT)
        self.timeScroll.sliderPressed.connect(preMove)


        #Slider for Z axis
        #self.layerScroll.setMaximum(self.nPos-1)
        def updateZ():
            self.imv.setImage(self.finalArray[self.layerScroll.sliderPosition()][:, :, self.timeScroll.sliderPosition()].T, autoRange=False, autoLevels=False)
            if self.ROIexists:
                update(self.roi)

        self.layerScroll.sliderMoved.connect(updateZ)


        #ROI slider functionality
        #Changes range
        #Main window slider functionality
        #Each functions the same as their ROI display counterparts
        def updateBottom():
            min = self.brightnessSlider.sliderPosition()-(self.contrastSlider.sliderPosition()/2)
            max = self.brightnessSlider.sliderPosition()+(self.contrastSlider.sliderPosition()/2)
            self.imv.setLevels(min, max)

        self.contrastSlider.sliderMoved.connect(updateBottom)
        self.contrastSlider.sliderPressed.connect(updateBottom)
        self.contrastSlider.sliderReleased.connect(updateBottom)

        def updateSide():
            min = self.brightnessSlider.sliderPosition()-(self.contrastSlider.sliderPosition()/2)
            max = self.brightnessSlider.sliderPosition()+(self.contrastSlider.sliderPosition()/2)
            self.imv.setLevels(min, max)

        self.brightnessSlider.sliderMoved.connect(updateBottom)
        self.brightnessSlider.sliderPressed.connect(updateBottom)
        self.brightnessSlider.sliderReleased.connect(updateBottom)

        updateBottom()
        #self.layerScroll.setSliderPosition((self.nPos-1)/2)
        #self.imageView.viewRect()
        #######################################################>>>>>>>>>NEW PAST HERE<<<<<<<<###################################################
        ####COCalculator functions####
        self.resetPlot.clicked.connect(self.Reset)
        self.createCSV.clicked.connect(self.CSVcreator)

        #Directory choice functionality

        def select():
            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.Directory)
            dialog.setWindowTitle("Enter Data Path")
            PathDicom = dialog.getExistingDirectory()
            print(PathDicom)
            if (PathDicom == "" or PathDicom == None):
                pass
            else:
                self.lstFilesDCM = []  # create an empty list
                for dirName, subdirList, fileList in os.walk(PathDicom):
                    for filename in fileList:
                        if ".dcm" in filename.lower():  # check whether the file's DICOM
                            if (dicom.read_file(os.path.join(dirName, filename))[0x18, 0x1030].value) == "PE Circ Time":
                                self.lstFilesDCM.append(os.path.join(dirName, filename))

                ChestCT = dicom.read_file(self.lstFilesDCM[0])

                ConstPixelDims = (int(ChestCT.Rows), int(ChestCT.Columns), len(self.lstFilesDCM))
                ConstPixelSpacing = (
                float(ChestCT.PixelSpacing[0]), float(ChestCT.PixelSpacing[1]), float(ChestCT.SliceThickness))
                out = ""
                out += "Pixel Dimentions: " + ConstPixelDims.__str__() + '\n' + "Pixel Spacing: " + ConstPixelSpacing.__str__() + '\n'

                self.accessionNum = dicom.read_file(self.lstFilesDCM[1])[0x8, 0x50].value
                self.seenPos = []  # List to hold possible positions
                self.seenTime = []  # List to hold possible times
                masterList = []  # List to hold data in filename, position, time format
                for temp in self.lstFilesDCM:
                    dc = dicom.read_file(temp)
                    if self.seenPos.__contains__(dc[0x20, 0x1041].value):
                        pass
                    else:
                        self.seenPos.append(dc[0x20, 0x1041].value)
                    if self.seenTime.__contains__(dc[0x8, 0x32].value):
                        pass
                    else:
                        self.seenTime.append(dc[0x8, 0x32].value)
                    masterList.append([temp, dc[0x20, 0x1041].value, dc[0x8, 0x32].value])
                if self.seenTime.__len__() > 25:
                    self.HUvalues.setRowCount(self.seenTime.__len__() + 1)
                    self.HUvalues.setVerticalScrollBarPolicy(2)
                temptime = sorted(self.seenTime)
                temp1 = float(temptime[1])
                temp2 = float(temptime[0])
                self.timeIntervalfloat = round(temp1 - temp2, 4)

                self.timeInterval.setPlainText(str(self.timeIntervalfloat))
                s = sorted(masterList, key=lambda x: (x[2]))
                s = sorted(s, key=lambda x: (x[1]))  # Sorted by position then by time (maybe)

                self.nPos = self.seenPos.__len__()
                self.nTime = self.seenTime.__len__()

                self.finalArray = []  # List holding all Dicom arrays
                for p in np.arange(0, self.nPos, 1):
                    ArrayDicom = np.zeros(ConstPixelDims, dtype=ChestCT.pixel_array.dtype)
                    for t in np.arange(0, self.nTime, 1):
                        fileDCM = s[t + p * self.nTime][0]
                        # read the file
                        ds = dicom.read_file(fileDCM)
                        # store the raw image data
                        ArrayDicom[:, :, t] = ds.pixel_array
                    self.finalArray.append(ArrayDicom)
                clearBASE()
                clearROI()
                self.timeScroll.setSliderPosition(0)
                self.layerScroll.setSliderPosition(0)
                self.imv.setImage(self.finalArray[self.layerScroll.sliderPosition()][:, :, self.timeScroll.sliderPosition()].T, autoRange=False, autoLevels=False)
                self.timeScroll.setMaximum(self.nTime - 1)
                self.layerScroll.setMaximum(self.nPos - 1)
                updateZ()

        self.DirSelect.clicked.connect(select)

        #Copy data to clipboard
        def copyClip():
            clipboard = QApplication.clipboard()
            text = "Accession Number: \t" + dicom.read_file(self.lstFilesDCM[0])[0x8, 0x50].value + "\n"
            text += "Baseline: \t" + self.baselineInput.toPlainText() + "\n"
            text += "Cardiac Output (L/min): \t" + self.cardiacOutput.toPlainText() + "\n"
            text += "Standard Error: \t" + self.standardError.toPlainText() + "\n"
            text += "Time to Peak: \t" + self.peakTime.toPlainText() + "\n"
            text += "Mean Transit Time (s): \t" + self.MTT.toPlainText() + "\n"
            text += "HU enhancement: \t"
            for i in np.arange(0, self.seenTime.__len__(), 1):
                text += str(self.HUvalues.item(i, 1).text()) + "\n\t"
            clipboard.setText(text)

        self.copier.clicked.connect(copyClip)

        #Tooltips
        self.clearBASE_btn.setToolTip("Remove Baseline ROI")
        self.clearROI_btn.setToolTip("Remove MPA ROIs")
        self.createROI_btn.setToolTip("Create MPA ROIs at next mouse click")
        self.baseROI_btn.setToolTip("Create Baseline ROI at next mouse click, on first time interval")
        self.resetPlot.setToolTip("Clear plot and data used to create it")
        self.calcAndPlot_btn.setToolTip("Create data from ROIs and plot")
        self.DirSelect.setToolTip("Select the directory that holds the desired data to change imageset")
        self.copier.setToolTip("Copy data read from images to the clipboard")
        self.createCSV.setToolTip("Create text file in CSV format")


    def ApplyChecker(self, parent = None):
        check = True
        error = ""
        # Time interval checks
        ############################################
        if(self.timeInterval.toPlainText() == ""):
            check = False
            error += "Time interval must be entered\n"
        else:
            try:
                ti = float(self.timeInterval.toPlainText())
                if ti <= 0:
                    error += "Time interval must be greater than zero\n"
                    check = False
            except ValueError:
                error += "Time interval must consist only of numbers"
                check = False
        # baseline checks
        #############################################
        if (self.baselineInput.toPlainText() == ""):
            check = False
            error += "Baseline must be entered\n"
        else:
            try:
                b = float(self.baselineInput.toPlainText())
            except ValueError:
                error += "Baseline must consist only of numbers\n"
                check = False
        # HUValues
        ##########################################
        try:
            a = []
            self.clearFocus()
            allRows = self.HUvalues.rowCount()
            for i in np.arange(0, allRows + 1, 1):
                temp = self.HUvalues.item(i, 1)
                if temp:
                    a.append(float(temp.text()))
            a = np.array(a)  # type float
        except ValueError:
            check = False
            error += "Table Inputs must consist only of numbers\n"
        # Final check
        #############################################
        if check == True:
            try:
                self.Apply()
            except:
                check = False
                error += "An unknown error occured, please ensure all data is correct"
        if check == False:
            # print("error box popup:\n", error
            msgBox = QMessageBox()
            msgBox.setText("There was an error:")
            msgBox.setInformativeText(error)
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setDefaultButton(QMessageBox.Ok)
            msgBox.exec_()

    def CSVcreator(self,parent = None):

        filename = ("csvfiles/"+self.accessionNum + "_" + str(datetime.datetime.today()))
        filename = filename.replace(" ", "_")
        filename = filename.replace(":", "-")
        filename = filename.split(".", 1)[0]
        filename = filename + ".csv"
        try:
            with open(filename, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=' ',
                                        quotechar=' ', quoting=csv.QUOTE_MINIMAL)

                self.clearFocus()
                allRows = self.HUvalues.rowCount()
                roiValues = ["ROI Values:"]
                for i in np.arange(0, allRows + 1, 1):
                    temp = self.HUvalues.item(i, 1)
                    if temp:
                        temp = self.HUvalues.item(i, 1).text()
                        roiValues.append(temp)
                roiValuesStr = ""
                for item in roiValues:
                    roiValuesStr += (str(item)+",")
                writer.writerow(roiValuesStr)
                writer.writerow(['Baseline:,',self.baselineInput.toPlainText()])
                writer.writerow(['HU/Iodine (HU/(mg/mL)):,', self.HUtoIodineConversion.toPlainText()])
                writer.writerow(['Time Interval (s):,', self.timeInterval.toPlainText()])
                writer.writerow(['Cardiac Output (L/min):,', self.cardiacOutput.toPlainText()])
                writer.writerow(['Time to peak:,',self.peakTime.toPlainText()])
                writer.writerow(['Mean Transit Time:,',self.MTT.toPlainText()])
                writer.writerow(['Area Under Curve:,',self.AUC.toPlainText()])
                writer.writerow(['Standard Error:,',self.standardError.toPlainText()])
                writer.writerow(['Alpha:,',self.alpha.toPlainText()])
                writer.writerow(['Beta:,',self.beta.toPlainText()])
                writer.writerow(['t0 (s):,',self.t0.toPlainText()])
                writer.writerow(['R Squared:,',self.rsquared.toPlainText()])
        except:
            msgBox = QMessageBox()
            msgBox.setText("There was an error:")
            msgBox.setInformativeText("An error occurred when trying to create a csv file")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setDefaultButton(QMessageBox.Ok)
            msgBox.exec_()

    def Apply(self, parent = None):
        self.createCSV.setEnabled(True)
        self.plotView.clear()
        a = []
        a_temp = []
        self.clearFocus()
        allRows = self.HUvalues.rowCount()
        for i in np.arange(0, allRows + 1, 1):
            temp = self.HUvalues.item(i, 1)
            if temp:
                a.append(float(temp.text()))
        a = np.array(a)  # type float
        b = float(self.baselineInput.toPlainText())
        self.patient.baseline = b
        self.patient.data = a - b
        self.patient.getCoeffs()
        self.patient.getR2()
        self.patient.getContData()
        self.patient.getStats()
        self.alpha.setPlainText(str(round(self.patient.alpha, 3)))
        self.beta.setPlainText(str(round(self.patient.beta, 3)))
        # self.t0.setPlainText(str(self.patient.)
        self.cardiacOutput.setPlainText(str(round(self.patient.CO, 3)))
        self.AUC.setPlainText(str(round(self.patient.AUC, 0)))
        self.rsquared.setPlainText(str(round(self.patient.R2, 3)))
        self.peakTime.setPlainText(str(round(self.patient.TTP, 3)))
        self.MTT.setPlainText(str(round(self.patient.MTT, 3)))
        # create array of xvalues based on time interval i/p, for plotting later
        t = self.timeInterval.toPlainText()  # type str
        x = np.arange(0, 100, float(t),
                      np.dtype(np.float))  # makes x-axis in terms of entered time intervals.(float i/p)
        xvalues = ([])
        includedx = self.patient.data.size
        for j in np.arange(0, includedx, 1):
            temp2 = x.item(j)
            xvalues.append(temp2)
        xvalues = np.array(xvalues)
        # xvalues = xvalues  # + self.patient.shift
        # estimate standard error of CO calculation with monte carlo simulation
        # first calculate residuals & st. dev.
        resids = np.zeros(self.patient.data.size, dtype=float)
        GVvalueDataSet = np.empty(self.patient.data.size, dtype=float)
        for k in np.arange(0, self.patient.data.size, 1):
            # temp3 = self.patient.data.size.item(k)
            GVvalue = self.patient.gammaFunc(xvalues[k], self.patient.A, self.patient.alpha, self.patient.beta)
            GVvalueDataSet[k] = GVvalue
            # GVvalueDataSet.append(GVvalue)
            residValue = self.patient.data[k] - GVvalue
            resids[k] = residValue
        # resids = np.array(resids)
        # GVvalueDataSet = np.array(GVvalue)
        residSD = np.asscalar(np.std(resids, dtype=float))
        # monte carlo
        mcLoops = 100
        fakeDataSet = np.empty(self.patient.data.size, dtype=float)
        fakeCOs = np.empty(mcLoops, dtype=float)
        self.times = np.arange(self.patient.shift, self.patient.shift + len(self.patient.data) * 2, 2)
        for m in np.arange(0, mcLoops, 1):
            for l in np.arange(0, self.patient.data.size, 1):
                # temp4 = self.patient.data.size.item(l)
                fakeDataPt = np.random.normal(0.0, residSD) + GVvalueDataSet[l]
                fakeDataSet[l] = fakeDataPt
            popt, pcov = curve_fit(self.patient.gammaFunc, self.times, fakeDataSet, maxfev=5000)
            mcA, mcAlpha, mcBeta = popt[0], popt[1], popt[2]
            if (mcAlpha < 0):
                print('curve fit in MC returned negative Alpha')
            mcContData = mcA * (self.patient.contTimes ** mcAlpha) * np.exp(-self.patient.contTimes / mcBeta)
            mcAUC = np.trapz([mcContData], x=[self.patient.contTimes])
            Imass = 0.3 * 350 * 75
            fakeCOs[m] = Imass / mcAUC * 24 * 60 / 1000
        fakeDataSD = np.std(fakeCOs)
        self.standardError.setPlainText(str(round(fakeDataSD, 3)))
        # plot with pyqtgraph
        # self.GraphicsView.setConfigOption('background', 'w')
        # self.GraphicsView.setConfigOption('foreground', 'k')
        # self.GraphicsView.setBackground('#0061ff')
        # self.GraphicsView.setConfigOption('foreground','#0061ff')
        self.plotView.plot(title=' ')
        self.legend = self.plotView.addLegend(size=(100, 40), offset=(0, 1))

        if self.legendExists:
            self.legend.scene().removeItem(self.legend)
        else:
            self.legendExists = True

        self.plotView.plot(self.patient.times, self.patient.data, name='Patient Data', pen=None, symbol='t',
                            symbolPen=None, symbolSize=5, symbolBrush=(204, 63, 12, 255))
        self.plotView.plot(self.patient.contTimes, self.patient.contData, name='Curve Fit',
                            pen=mkPen('b', width=1))

        self.plotView.setLabel('left', "Enhancement (HU)")
        self.plotView.setLabel('bottom', "Time (s)")
        self.plotView.viewRect()




    def Reset(self, parent = None):
        self.HUvalues.clear()
        self.alpha.clear()
        self.beta.clear()
        self.t0.clear()
        self.rsquared.clear()
        self.AUC.clear()
        self.cardiacOutput.clear()
        self.peakTime.clear()
        self.MTT.clear()
        self.standardError.clear()
        self.baselineInput.clear()
        self.patient.number = 0
        self.patient.data = 0
        self.patient.baseline = 0
        self.patient.shift = 0
        self.patient.A = 0
        self.patient.alpha = 0
        self.patient.Beta = 0
        self.patient.times = 0
        self.patient.fitdata = 0
        self.patient.R2 = 0
        self.patient.contTimes = 0
        self.patient.contData = 0
        self.patient.AUC = 0
        self.patient.CO = 0
        self.plotView.clear()
        self.timeInterval.setPlainText(str(self.timeIntervalfloat))
        self.createCSV.setEnabled(False)




    #Changes the left click to set the coordinates to create an ROI

    def mousePressEvent(self, a0: QtGui.QMouseEvent):
        buttons = qApp.mouseButtons()
        QApplication.restoreOverrideCursor()
        if(buttons==QtCore.Qt.LeftButton):
            #print("Middle Button Clicked!")
            x = a0.x()
            y = a0.y()
            #print(x, y)
            self.xCoordTxt.setPlainText((((x)-10)-self.spinBoxROI.value()/2).__str__())
            self.yCoordTxt.setPlainText((((y)-10)-self.spinBoxROI.value()/2).__str__())
            if(self.createROI_btn.isDown()):
                if self.ROIexists == False:
                    self.ROIexists = True
                    # Creates the ROI list
                    self.roi.setParentItem(self.imv.getView())
                    # roi.setPos(100,100)
                    self.roi.setPen(mkPen('y', width=3, style=QtCore.Qt.DashLine))
                    coordsCorrect = True
                    try:
                        temp = float(self.xCoordTxt.toPlainText())
                    except:
                        coordsCorrect = False
                    if coordsCorrect:
                        self.roi.setPos(float(self.xCoordTxt.toPlainText()), float(self.yCoordTxt.toPlainText()))
                        for i in np.arange(0, self.nTime, 1):
                            self.roiList.append(self.roi.saveState())
                        # Generates second image and output from ROI data
                        ROIarray = self.roi.getArrayRegion(self.finalArray[self.layerScroll.sliderPosition()][:, :,
                                                           self.timeScroll.sliderPosition()].T, self.imv.getImageItem())
                        self.MPAmean.setPlainText(str(ROIarray.mean()))
                        np.fliplr(ROIarray)

                    if not coordsCorrect:
                        msgBox = QMessageBox()
                        msgBox.setText("There was an error:")
                        msgBox.setInformativeText("Ensure that coordinates for ROI are correct and try again.")
                        msgBox.setStandardButtons(QMessageBox.Ok)
                        msgBox.setDefaultButton(QMessageBox.Ok)
                        msgBox.exec_()
                self.createROI_btn.setDown(False)
            elif(self.baseROI_btn.isDown()):
                if self.BASEexists == False:
                    self.BASEexists = True
                    self.BASEroi.setParentItem(self.imv.getView())
                    self.BASEroi.setPen(mkPen('c', width=3, style=QtCore.Qt.DashLine))
                    self.BASEroi.setSize(self.spinBoxROI.value(), self.spinBoxROI.value())
                    coordsCorrect = True
                    try:
                        temp = float(self.xCoordTxt.toPlainText())
                    except:
                        coordsCorrect = False
                    if coordsCorrect:
                        self.BASEroi.setPos(float(self.xCoordTxt.toPlainText()), float(self.yCoordTxt.toPlainText()))
                        self.BASEroiSAVE = self.BASEroi.saveState()
                        self.BASEmean.setPlainText(str(self.BASEroi.getArrayRegion(self.finalArray[self.layerScroll.sliderPosition()][:, :, 0].T, self.imv.getImageItem()).mean()))
                    if not coordsCorrect:
                        msgBox = QMessageBox()
                        msgBox.setText("There was an error:")
                        msgBox.setInformativeText("Ensure that coordinates for baseline are correct and try again.")
                        msgBox.setStandardButtons(QMessageBox.Ok)
                        msgBox.setDefaultButton(QMessageBox.Ok)
                        msgBox.exec_()
                self.baseROI_btn.setDown(False)