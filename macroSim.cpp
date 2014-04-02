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

#define ITOM_IMPORT_API
#define ITOM_IMPORT_PLOTAPI

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

#include "DataObject/dataObjectFuncs.h"

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
    filter = new FilterDef(MacroSim::runMacroSimRayTrace, MacroSim::runMacroSimRayTraceParams, "starts a MacroSim-Raytrace", ito::AddInAlgo::catNone, ito::AddInAlgo::iNotSpecified);
    m_filterList.insert("runMacroSimRayTrace", filter);
    filter = new FilterDef(MacroSim::simConfRawSignal, MacroSim::simConfRawSignalParams, "starts a simulation of the raw signal of a confocal point sensor", ito::AddInAlgo::catNone, ito::AddInAlgo::iNotSpecified);
    m_filterList.insert("simConfRawSignal", filter);
    filter = new FilterDef(MacroSim::simConfSensorSignal, MacroSim::simConfSensorSignalParams, "starts a simulation of a confocal point sensor", ito::AddInAlgo::catNone, ito::AddInAlgo::iNotSpecified);
    m_filterList.insert("simConfSensorSignal", filter);
    filter = new FilterDef(MacroSim::calcCuFFT, MacroSim::calcCuFFTParams, "calculates a FFT on a CUDA GPU", ito::AddInAlgo::catNone, ito::AddInAlgo::iNotSpecified);
    m_filterList.insert("calcCuFFT", filter);


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
ito::RetVal MacroSim::runMacroSimRayTrace(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    
	// call to macrosim-runMacroSimRayTrace()
	if ((*paramsMand).size() != 2)
	{
		retval = ito::RetVal(ito::retError,0,"Error in runMacroSimRayTrace. functions takes exactly two parameter");
		return retval;
	}
	char *l_pSceneChar = (*paramsMand)[0].getVal<char*>();
	ito::DataObject *l_pDataObject = (*paramsMand)[1].getVal<ito::DataObject*>();
	ito::DataObject *t_pDataObject=new ito::DataObject();

	void* l_pField;
	ItomFieldParams l_fieldParams;

	if ( (l_pSceneChar==NULL) )
	{
		retval = ito::retError;
		return retval;
	}

	MacroSimTracer l_tracer;

	bool ret=l_tracer.runMacroSimRayTrace(l_pSceneChar, &l_pField, &l_fieldParams, NULL, NULL);
	if (ret)
	{
		if (createDataObjectFromMacroSimResult(l_fieldParams, t_pDataObject, l_pField))
			*l_pDataObject=*t_pDataObject;
	}
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
ito::RetVal MacroSim::runMacroSimRayTraceParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    ito::Param param;
	retval += ito::checkParamVectors(paramsMand,paramsOpt,paramsOut);
	if(retval.containsError()) return retval;

	paramsMand->clear();
    paramsMand->append(ito::Param("sceneFile", ito::ParamBase::String | ito::ParamBase::In, "", tr("complete filename of the scene prescription file").toLatin1().data()) );
    paramsMand->append(ito::Param("resultPtr", ito::ParamBase::DObjPtr | ito::ParamBase::In | ito::ParamBase::Out, NULL, tr("pointer to the dataObject where the result will be saved").toLatin1().data()) );

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
ito::RetVal MacroSim::simConfRawSignal(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    
	// call to macrosim-runMacroSimRayTrace()
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

	bool ret=l_tracer.runConfRawSigSim(l_confPointParams, &l_pResult);

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

    delete l_pResult;
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
ito::RetVal MacroSim::simConfRawSignalParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    ito::Param param;
	retval += ito::checkParamVectors(paramsMand,paramsOpt,paramsOut);
	if(retval.containsError()) return retval;

	paramsMand->clear();
    paramsMand->append(ito::Param("params", ito::ParamBase::DoubleArray | ito::ParamBase::In, 0, tr("vector holding the parameters of the simulation. The format is as follows: gridWidth, , number of sample points along grid, magnification, NA, number of scan points in x, number of scan points in y, number of scan points in z, scan step in x, scan step in y, scan step in z, wavelength, vector containing 16 zernike coefficients").toLatin1().data()) );
	paramsMand->append(ito::Param("resultPtr", ito::ParamBase::DObjPtr | ito::ParamBase::In | ito::ParamBase::Out, NULL, tr("pointer to the dataObject where the result will be saved").toLatin1().data()) );

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
ito::RetVal MacroSim::simConfSensorSignal(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    
	// call to macrosim-runMacroSimRayTrace()
	if ((*paramsMand).size() != 3)
	{
		retval = ito::retError;
		return retval;
	}
	ito::DataObject *l_pParamsDataObj = (*paramsMand)[0].getVal<ito::DataObject*>();
    ito::DataObject *l_pObjParamsDataObj = (*paramsMand)[1].getVal<ito::DataObject*>();
	ito::DataObject *l_pDataObject = (*paramsMand)[2].getVal<ito::DataObject*>();

	ConfPoint_Params l_confPointParams;
    l_confPointParams.gridWidth=l_pParamsDataObj->at<double>(0,0);
	l_confPointParams.n=l_pParamsDataObj->at<double>(0,1);
	l_confPointParams.magnif=l_pParamsDataObj->at<double>(0,2);
	l_confPointParams.NA=l_pParamsDataObj->at<double>(0,3);
	l_confPointParams.scanNumber.x=l_pParamsDataObj->at<double>(0,4);
	l_confPointParams.scanNumber.y=l_pParamsDataObj->at<double>(0,5);
	l_confPointParams.scanNumber.z=l_pParamsDataObj->at<double>(0,6);
	l_confPointParams.scanStep.x=l_pParamsDataObj->at<double>(0,7);
	l_confPointParams.scanStep.y=l_pParamsDataObj->at<double>(0,8);
	l_confPointParams.scanStep.z=l_pParamsDataObj->at<double>(0,9);
	l_confPointParams.wvl=l_pParamsDataObj->at<double>(0,10);
	l_confPointParams.apodisationRadius=l_pParamsDataObj->at<double>(0,11);
    l_confPointParams.pAberrVec[0]=l_pParamsDataObj->at<double>(0,12);
    l_confPointParams.pAberrVec[1]=l_pParamsDataObj->at<double>(0,13);
    l_confPointParams.pAberrVec[2]=l_pParamsDataObj->at<double>(0,14);
    l_confPointParams.pAberrVec[3]=l_pParamsDataObj->at<double>(0,15);
    l_confPointParams.pAberrVec[4]=l_pParamsDataObj->at<double>(0,16);
    l_confPointParams.pAberrVec[5]=l_pParamsDataObj->at<double>(0,17);
    l_confPointParams.pAberrVec[6]=l_pParamsDataObj->at<double>(0,18);
    l_confPointParams.pAberrVec[7]=l_pParamsDataObj->at<double>(0,19);
    l_confPointParams.pAberrVec[8]=l_pParamsDataObj->at<double>(0,20);
    l_confPointParams.pAberrVec[9]=l_pParamsDataObj->at<double>(0,21);
    l_confPointParams.pAberrVec[10]=l_pParamsDataObj->at<double>(0,22);
    l_confPointParams.pAberrVec[11]=l_pParamsDataObj->at<double>(0,23);
    l_confPointParams.pAberrVec[12]=l_pParamsDataObj->at<double>(0,24);
    l_confPointParams.pAberrVec[13]=l_pParamsDataObj->at<double>(0,25);
    l_confPointParams.pAberrVec[14]=l_pParamsDataObj->at<double>(0,26);
    l_confPointParams.pAberrVec[15]=l_pParamsDataObj->at<double>(0,27);

    ConfPointObject_Params l_confPointObjectParams;
    l_confPointObjectParams.A=l_pObjParamsDataObj->at<double>(0,0);
    l_confPointObjectParams.kN=l_pObjParamsDataObj->at<double>(0,1);

	MacroSimTracer l_tracer;
	double* l_pResult;

	bool ret=l_tracer.runConfSensorSigSim(l_confPointParams, l_confPointObjectParams, &l_pResult);

	if (ret)
	{
		// create continous dataObject
		//*l_pDataObject = ito::DataObject(l_confPointParams.scanNumber.x, l_confPointParams.scanNumber.y, ito::tFloat64);
        int sizes[2];
        sizes[0]=l_confPointParams.scanNumber.x;
        sizes[1]=l_confPointParams.scanNumber.y;
        *l_pDataObject = ito::DataObject(2, sizes, ito::tFloat64, 1);
		cv::Mat *MAT1;
		MAT1 = (cv::Mat*)(l_pDataObject->get_mdata()[l_pDataObject->seekMat(0)]);
		memcpy(MAT1->ptr(0), l_pResult, l_confPointParams.scanNumber.x*l_confPointParams.scanNumber.y*sizeof(double));
		// set data object tags
		l_pDataObject->setAxisOffset(0, -l_confPointParams.scanStep.x*l_confPointParams.scanNumber.x/2);
		l_pDataObject->setAxisOffset(1, -l_confPointParams.scanStep.y*l_confPointParams.scanNumber.y/2);
		l_pDataObject->setAxisScale(0, l_confPointParams.scanStep.x);
		l_pDataObject->setAxisScale(1, l_confPointParams.scanStep.y);
		l_pDataObject->setAxisUnit(0,"um");
		l_pDataObject->setAxisUnit(1,"um");
        l_pDataObject->setValueUnit("um");
		
		// create data object from l_pField and push it to outVals
	}
	else
		retval = ito::retError;

    delete l_pResult;
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
ito::RetVal MacroSim::simConfSensorSignalParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    ito::Param param;
	retval += ito::checkParamVectors(paramsMand,paramsOpt,paramsOut);
	if(retval.containsError()) return retval;

	paramsMand->clear();
	paramsMand->append(ito::Param("paramsSensor", ito::ParamBase::DObjPtr | ito::ParamBase::In, 0, tr("pointer to the dataObject holding the parameters of the confocal sensor. The format is as follows: gridWidth, , number of sample points along grid, magnification, NA, number of scan points in x, number of scan points in y, number of scan points in z, scan step in x, scan step in y, scan step in z, wavelength, apodisation radius, vector containing 16 zernike coefficients").toLatin1().data()) );
    paramsMand->append(ito::Param("paramsObj", ito::ParamBase::DObjPtr | ito::ParamBase::In, 0, tr("pointer to the dataObject holding the parameters of the sinusoidal object. The format is as follows: amplitude, wavenumber").toLatin1().data()) );
	paramsMand->append(ito::Param("resultPtr", ito::ParamBase::DObjPtr | ito::ParamBase::In | ito::ParamBase::Out, NULL, tr("pointer to the dataObject where the result will be saved").toLatin1().data()) );

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
ito::RetVal MacroSim::calcCuFFT(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    
	if ((*paramsMand).size() != 1)
	{
		retval = ito::retError;
        retval.appendRetMessage("function needs 3 parameters");
		return retval;
	}
	ito::DataObject *l_pDataObj = (*paramsMand)[0].getVal<ito::DataObject*>();

    retval += ito::dObjHelper::verifyDataObjectType(l_pDataObj, "dObjIn", 1, ito::tComplex128);

    if(retval.containsError())
    {
        return retval;
    }

    unsigned int l_dims=l_pDataObj->getDims();
    if (l_dims != 2)
    {
        retval = ito::retError;
        retval.appendRetMessage("input field must be two dimensional");
    }

    ito::int32 xSize = static_cast<ito::int32>(l_pDataObj->getSize(l_dims - 1));
    ito::int32 ySize = static_cast<ito::int32>(l_pDataObj->getSize(l_dims - 2));

    if (xSize == 1 || ySize == 1)
    {
        return ito::RetVal(ito::retError, 0, tr("Error: field dimensions must not be 1xN or Nx1.").toLatin1().data());
    }

    cuDoubleComplex *l_pUin=(cuDoubleComplex*)((cv::Mat*)l_pDataObj->get_mdata()[l_pDataObj->seekMat(0)])->ptr(0);

	MacroSimTracer l_tracer;
    
	bool ret=l_tracer.calcCuFFT(l_pUin, xSize, ySize);

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
ito::RetVal MacroSim::calcCuFFTParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut)
{
    ito::RetVal retval = ito::retOk;
    ito::Param param;
	retval += ito::checkParamVectors(paramsMand,paramsOpt,paramsOut);
	if(retval.containsError()) return retval;

	paramsMand->clear();
    paramsMand->append(ito::Param("Uin", ito::ParamBase::DObjPtr | ito::ParamBase::In | ito::ParamBase::Out, NULL, tr("pointer to the dataObject of the input. The result will also be stored here, resulting in an in place operation").toLatin1().data()) );

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

//----------------------------------------------------------------------------------------------------------------------------------
/** create an itom dataObject from the result of MacroSim
*	@param [in]	ItomFieldParams &fieldParams, void *MacroSimResult
*	@param [in]	ito::DataObject &dataObject
*
*/
bool createDataObjectFromMacroSimResult(ItomFieldParams &fieldParams, ito::DataObject *dataObject, void *MacroSimResult)
{
	if (MacroSimResult)
	{
		double *l_ptr=static_cast<double*>(MacroSimResult);
		// copy the resulting field into a dataObject
		// create new dataObject according to parameters of resulting field
		cv::Mat *MAT1;
		switch (fieldParams.type)
		{
		case INTFIELD:
			//*dataObject = ito::DataObject(fieldParams.nrPixels[1], fieldParams.nrPixels[0], ito::tFloat64);
			*dataObject = ito::DataObject(fieldParams.nrPixels[2], fieldParams.nrPixels[1], fieldParams.nrPixels[0], ito::tFloat64);
			for (unsigned int jz=0; jz<fieldParams.nrPixels[2]; jz++)
			{
				size_t planeID=dataObject->seekMat(jz);
				MAT1 = (cv::Mat*)(dataObject->get_mdata()[dataObject->seekMat(jz)]);
				//memcpy(MAT1->ptr(0), MacroSimResult, fieldParams.nrPixels[1]*fieldParams.nrPixels[0]*sizeof(double));
				memcpy(MAT1->ptr(0), l_ptr+jz*fieldParams.nrPixels[0]*fieldParams.nrPixels[1], fieldParams.nrPixels[1]*fieldParams.nrPixels[0]*sizeof(double));
			}
			// set data object tags
			dataObject->setAxisOffset(0, fieldParams.MTransform[3]);//-floorf(l_fieldParams.nrPixels[0]/2)*l_fieldParams.scale[0]);
			dataObject->setAxisOffset(1, fieldParams.MTransform[7]);//-floorf(l_fieldParams.nrPixels[0]/2)*l_fieldParams.scale[1]);
			dataObject->setAxisOffset(2, fieldParams.MTransform[11]);
			dataObject->setAxisScale(0, fieldParams.scale[0]);
			dataObject->setAxisScale(1, fieldParams.scale[1]);
			dataObject->setAxisScale(2, fieldParams.scale[2]);
			dataObject->setAxisUnit(0,"mm");
			dataObject->setAxisUnit(1,"mm");
			dataObject->setTag("wvl", fieldParams.lambda);
			dataObject->setTag("title", "Intensity Field");
			dataObject->setValueDescription("intensity");
			dataObject->setValueUnit("a.u.");
			dataObject->addToProtocol("Traced with ito macro sim");

			break;
		//case SCALARFIELD:
		//	*dataObject = ito::DataObject(fieldParams.nrPixels[2], fieldParams.nrPixels[1], fieldParams.nrPixels[0], ito::tComplex64);
		//	break;
		default:
			cout << "error in TracerThread.runMacroSimRayTrace(): MacroSim returned a field type, that cannot be converted to itom::dataObject yet..." << endl;
			return false; // signal error
			break;
		}
	}
	else
		return false;
	return true;
};
