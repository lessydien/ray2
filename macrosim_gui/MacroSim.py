from itom import *
import numpy as np

def plotResult(resultObject):
    global result
    result=resultObject;
    plot(result,'itom2dQwtPlot')

if(__name__ == "__main__"):
#    messPlanAssist = MessplanAssistent()
#    messPlanAssist.show()
    addButton("MacroSim", "start MacroSim","start_macroSim()","")
    
def start_macroSim():
    global mainWidget
    mainWidget=ui.createNewPluginWidget("MacroSim_MainWin")
    mainWidget.show()
    mainWidget.connect("simulationFinished(ito::DataObject)", plotResult)
