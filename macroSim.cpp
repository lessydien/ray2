/***********************************************************************
 This file is part of ITO-MacroSim.

    ITO-MacroSim is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ITO-MacroSim is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
************************************************************************/

#include "mainWinMacroSim.h"
#include "macroSim.h"
#include "common/helperCommon.h"

#include <QtCore/QtPlugin>

#include "TracerThread.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "pluginVersion.h"
#include "common/apiFunctionsInc.h"

void ** ITOM_API_FUNCS=NULL;

//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MacroSimInterface::getAddInInst(ito::AddInBase **addInInst)
{
    MacroSim* newInst = new MacroSim();
    newInst->setBasePlugin(this);
    *addInInst = qobject_cast<ito::AddInBase*>(newInst);
	m_InstList.append(*addInInst);

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
ito::RetVal MacroSimInterface::closeThisInst(ito::AddInBase **addInInst)
{
    if (*addInInst)
    {
        delete ((MacroSim *)*addInInst);
        int idx = m_InstList.indexOf(*addInInst);
        m_InstList.removeAt(idx);
    }

    return ito::retOk;
}

//----------------------------------------------------------------------------------------------------------------------------------
MacroSimInterface::MacroSimInterface()
{
    m_type = ito::typeAlgo;
    setObjectName("MacroSim");
    
    m_description = QObject::tr("Optical simulation suite");
    m_detaildescription = QObject::tr("Our own little package for simulating optical systems via GPU accelerated raytracing or scalar wave optics.");
    m_author = "F. Mauch, ITO, University Stuttgart";
    m_version = (PLUGIN_VERSION_MAJOR << 16) + (PLUGIN_VERSION_MINOR << 8) + PLUGIN_VERSION_PATCH;
    m_minItomVer = MINVERSION;
    m_maxItomVer = MAXVERSION;
    m_license = QObject::tr("GNU GPL 3.0");
    m_aboutThis = QObject::tr("N.A.");     
    
}

//----------------------------------------------------------------------------------------------------------------------------------
MacroSimInterface::~MacroSimInterface()
{
    m_initParamsMand.clear();
    m_initParamsOpt.clear();
}

//----------------------------------------------------------------------------------------------------------------------------------
Q_EXPORT_PLUGIN2(MacroSim, MacroSimInterface)


//----------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------
MacroSim::MacroSim() : AddInAlgo()
{
}


//----------------------------------------------------------------------------------------------------------------------------------
/** initialize filter functions within this addIn
*	@param [in]	paramsMand	mandatory parameters that have to passed to the addIn on initialization
*	@param [in]	paramsOpt	optional parameters that can be passed to the addIn on initialization
*	@return					retError in case of an error
*
*	Here are the filter functions defined that are available through this addIn.
*	These are:
*       - filterName    description for this filter
*   
*   This plugin additionally makes available the following widgets, dialogs...:
*       - dialogName    description for this widget
*/
ito::RetVal MacroSim::init(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ItomSharedSemaphore *waitCond)
{
    ito::RetVal retval = ito::retOk;

	// fill API-pointer
	if (ITOM_API_FUNCS==NULL)
		ITOM_API_FUNCS=m_pBasePlugin->m_apiFunctionsBasePtr;

    ItomSharedSemaphoreLocker locker(waitCond);

    FilterDef *filter = NULL;
    AlgoWidgetDef *widget = NULL;
    
    //specify filters here, example:
    filter = new FilterDef(MacroSim::runSimulation, MacroSim::runSimulationParams, "starts a simulation", ito::AddInAlgo::catNone, ito::AddInAlgo::iNotSpecified);
    m_filterList.insert("runSimulation", filter);
    filter = new FilterDef(MacroSim::simConfPointSensor, MacroSim::simConfPointSensorParams, "starts a simulation of a confocal point sensor", ito::AddInAlgo::catNone, ito::AddInAlgo::iNotSpecified);
    m_filterList.insert("simConfPointSensor", filter);


    //specify dialogs, main-windows, widgets... here, example:
	widget = new AlgoWidgetDef(MacroSim::dialog, MacroSim::dialogParams, "MacroSim-GUI", ito::AddInAlgo::catNone, ito::AddInAlgo::iNotSpecified);
    m_algoWidgetList.insert("MacroSim_MainWin", widget);

    if (waitCond) 
    {
        waitCond->returnValue = retval;
        waitCond->release();
    }

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** short description for this filter
*	@param [in]	paramsMand	mandatory parameters
*	@param [in]	paramsOpt	optional parameters
*
*	longer description for this filter
*/
ito::RetVal MacroSim::runSimulation(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    
	// call to macrosim-runSimulation()
	if ((*paramsMand).size() != 3)
	{
		retval = ito::retError;
		return retval;
	}
	char *l_pSceneChar = (*paramsMand)[0].getVal<char*>();
	ito::DataObject *l_pDataObject = (*paramsMand)[0].getVal<ito::DataObject*>();

	void* l_pField;
	ItomFieldParams l_fieldParams;

	MacroSimTracerParams *l_pParams=reinterpret_cast<MacroSimTracerParams*>((*paramsMand)[2].getVal<void*>());

	if ( (l_pSceneChar==NULL) )
	{
		retval = ito::retError;
		return retval;
	}
//	string l_sceneStr=m_scene.toAscii();
//	char* sceneChar=new char [l_sceneStr.size()+1];
//	sceneChar=strcpy(sceneChar,l_sceneStr.c_str());

	MacroSimTracerParams l_simParams;

	MacroSimTracer l_tracer;

	bool ret=l_tracer.runSimulation(l_pSceneChar, &l_pField, &l_fieldParams, l_simParams, NULL, NULL);
	double* testVal=static_cast<double*>(l_pField);

	// create data object from l_pField and push it to outVals
   
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** parameters for calling the corresponding filter
*	@param [in]	paramsMand	mandatory parameters for calling the corresponding filter
*	@param [in]	paramsOpt	optional parameters for calling the corresponding filter
*
*	mand. Params:
*		- describe the mandatory parameters here (list)
*
*   opt. Params:
*       - describe the optional parameter here (list)
*/
ito::RetVal MacroSim::runSimulationParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    ito::Param param;
	retval += ito::checkParamVectors(paramsMand,paramsOpt,paramsOut);
	if(retval.containsError()) return retval;

	paramsMand->clear();
	paramsMand->append(ito::Param("sceneFile", ito::ParamBase::String, "", "complete filename of the scene prescription file") );
	paramsMand->append(ito::Param("resultPtr", ito::ParamBase::DObjPtr, NULL, "pointer to the dataObject where the result will be saved") );
	paramsMand->append(ito::Param("paramPtr", ito::ParamBase::Pointer, NULL, "pointer of type MacroSimTracerParams to the memory where the params of the simulation are stored") );

	paramsOpt->clear();

    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** short description for this filter
*	@param [in]	paramsMand	mandatory parameters
*	@param [in]	paramsOpt	optional parameters
*
*	longer description for this filter
*/
ito::RetVal MacroSim::simConfPointSensor(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    
	// call to macrosim-runSimulation()
	if ((*paramsMand).size() != 2)
	{
		retval = ito::retError;
		return retval;
	}
	double *l_paramsVec = (*paramsMand)[0].getVal<double*>();
	ito::DataObject *l_pDataObject = (*paramsMand)[1].getVal<ito::DataObject*>();

	ConfPoint_Params l_confPointParams;
	l_confPointParams.gridWidth=l_paramsVec[0];
	l_confPointParams.n=l_paramsVec[1];
	l_confPointParams.magnif=l_paramsVec[2];
	l_confPointParams.NA=l_paramsVec[3];
	l_confPointParams.scanNumber.x=l_paramsVec[4];
	l_confPointParams.scanNumber.y=l_paramsVec[5];
	l_confPointParams.scanNumber.z=l_paramsVec[6];
	l_confPointParams.scanStep.x=l_paramsVec[7];
	l_confPointParams.scanStep.y=l_paramsVec[8];
	l_confPointParams.scanStep.z=l_paramsVec[9];
	l_confPointParams.wvl=l_paramsVec[10];
	l_confPointParams.apodisationRadius=l_paramsVec[11];
	memcpy(&(l_confPointParams.pAberrVec[0]), &(l_paramsVec[12]), 16*sizeof(double));

	MacroSimTracer l_tracer;
	double* l_pResult;

	bool ret=l_tracer.runConfPointSensorSim(l_confPointParams, &l_pResult);

	if (ret)
	{
		// create continous dataObject
		*l_pDataObject = ito::DataObject(l_confPointParams.scanNumber.x, l_confPointParams.scanNumber.y, l_confPointParams.scanNumber.z, ito::tFloat64, 1);
		cv::Mat *MAT1;
		MAT1 = (cv::Mat*)(l_pDataObject->get_mdata()[l_pDataObject->seekMat(0)]);
		memcpy(MAT1->ptr(0), l_pResult, l_confPointParams.scanNumber.x*l_confPointParams.scanNumber.y*l_confPointParams.scanNumber.z*sizeof(double));
		// set data object tags
		l_pDataObject->setAxisOffset(0, -l_confPointParams.scanStep.x*l_confPointParams.scanNumber.x/2);
		l_pDataObject->setAxisOffset(1, -l_confPointParams.scanStep.y*l_confPointParams.scanNumber.y/2);
		l_pDataObject->setAxisOffset(2, -l_confPointParams.scanStep.z*l_confPointParams.scanNumber.z/2);
		l_pDataObject->setAxisScale(0, l_confPointParams.scanStep.x);
		l_pDataObject->setAxisScale(1, l_confPointParams.scanStep.y);
		l_pDataObject->setAxisScale(2, l_confPointParams.scanStep.z);
		l_pDataObject->setAxisUnit(0,"um");
		l_pDataObject->setAxisUnit(1,"um");
		l_pDataObject->setAxisUnit(2,"um");

		// create data object from l_pField and push it to outVals
	}
	else
		retval = ito::retError;
    return retval;
}

//----------------------------------------------------------------------------------------------------------------------------------
/** parameters for calling the corresponding filter
*	@param [in]	paramsMand	mandatory parameters for calling the corresponding filter
*	@param [in]	paramsOpt	optional parameters for calling the corresponding filter
*
*	mand. Params:
*		- test parameter1
*
*   opt. Params:
*       - describe the optional parameter here (list)
*/
ito::RetVal MacroSim::simConfPointSensorParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    ito::Param param;
	retval += ito::checkParamVectors(paramsMand,paramsOpt,paramsOut);
	if(retval.containsError()) return retval;

	paramsMand->clear();
	paramsMand->append(ito::Param("params", ito::ParamBase::DoubleArray, 0, "vector holding the parameters of the simulation. The format is as follows: gridWidth, , number of sample points along grid, magnification, NA, number of scan points in x, number of scan points in y, number of scan points in z, scan step in x, scan step in y, scan step in z, wavelength, vector containing 16 zernike coefficients") );
	paramsMand->append(ito::Param("resultPtr", ito::ParamBase::DObjPtr, NULL, "pointer to the dataObject where the result will be saved") );

	paramsOpt->clear();

    return retval;
}
//----------------------------------------------------------------------------------------------------------------------------------
/** short description for this widget
*	@param [in]	paramsMand	mandatory parameters
*	@param [in]	paramsOpt	optional parameters
*
*	longer description for this widget
*/
QWidget* MacroSim::dialog(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::RetVal &retValue)
{
    retValue += ito::retOk;
    
    //example:
	MainWinMacroSim *mainWin = new MainWinMacroSim(NULL);
	return qobject_cast<QWidget*>(mainWin);
}

//----------------------------------------------------------------------------------------------------------------------------------
/** parameters for calling dialog
*	@param [in]	paramsMand	mandatory parameters for calling dialog
*	@param [in]	paramsOpt	optional parameters for calling dialog
*
*	mand. Params:
*		- describe the mandatory parameters here (list)
*
*   opt. Params:
*       - describe the optional parameter here (list)
*/
ito::RetVal MacroSim::dialogParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    ito::Param param;

    if (!paramsMand)
    {
        retval = ito::RetVal(ito::retError, 0, "uninitialized vector for mandatory parameters!");
    }
    else if (!paramsOpt)
    {
        retval = ito::RetVal(ito::retError, 0, "uninitialized vector for optional parameters!");
    }
    else
    {
        //mandatory
        /*param = ito::Param("dataObject", ito::Param::typeDObjPtr, NULL, "description");
        paramsMand->append(param);
        param = ito::Param("doubleValue", ito::Param::typeDouble, 0.0, 65535.0, 10.0, "double value between 0.0 and 65535.0, default: 10.0");
        paramsMand->append(param);*/

        //optional
        /*param = ito::Param("integerValue", ito::Param::typeInt, 0.0, 65535.0, 65535.0, "integer value beween 0 and 65535, default: 65535");
        paramsOpt->append(param);*/
    }

    return retval;
}
