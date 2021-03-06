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

#ifndef MACROSIM_H
#define MACROSIM_H

#include "common/addInInterface.h"
#include "MacroSimLib.h"

bool createDataObjectFromMacroSimResult(ItomFieldParams &fieldParams, ito::DataObject *dataObject, void *MacroSimResult);

//----------------------------------------------------------------------------------------------------------------------------------
/** @class MacroSimInterface
*   @brief short description
*
*   AddIn Interface for the MacroSim class s. also \ref MacroSim
*/
class MacroSimInterface : public ito::AddInInterfaceBase
{
    Q_OBJECT
#if QT_VERSION >=  QT_VERSION_CHECK(5,0,0)
    Q_PLUGIN_METADATA(IID "ito.AddInInterfaceBase" )
#endif
    Q_INTERFACES(ito::AddInInterfaceBase)
    PLUGIN_ITOM_API

    protected:

    public:
        MacroSimInterface();
        ~MacroSimInterface();
        ito::RetVal getAddInInst(ito::AddInBase **addInInst);

    private:
        ito::RetVal closeThisInst(ito::AddInBase **addInInst);
};

//----------------------------------------------------------------------------------------------------------------------------------
/** @class MacroSim
*   @brief short description
*
*   longer description
*/
class MacroSim : public ito::AddInAlgo
{
    Q_OBJECT

    protected:
        MacroSim();
        ~MacroSim() {};

    public:
        friend class MacroSimInterface;
        
        static QWidget* dialog(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ito::RetVal &retValue);
		static ito::RetVal dialogParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);
        
        static ito::RetVal runMacroSimRayTrace(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut);
        static ito::RetVal runMacroSimRayTraceParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);
        static ito::RetVal simConfRawSignal(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut);
        static ito::RetVal simConfRawSignalParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);
        static ito::RetVal simConfSensorSignal(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut);
        static ito::RetVal simConfSensorSignalParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);
        static ito::RetVal calcCuFFT(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, QVector<ito::ParamBase> *paramsOut);
        static ito::RetVal calcCuFFTParams(QVector<ito::Param> *paramsMand, QVector<ito::Param> *paramsOpt, QVector<ito::Param> *paramsOut);

    private:		

    public slots:
		ito::RetVal init(QVector<ito::ParamBase> *paramsMand, QVector<ito::ParamBase> *paramsOpt, ItomSharedSemaphore *waitCond = NULL);
        ito::RetVal close(ItomSharedSemaphore *waitCond)
		{ 
			ItomSharedSemaphoreLocker locker(waitCond);
			waitCond->returnValue = ito::retOk;
			if(waitCond) waitCond->release();
			return ito::retOk; 
		};
};

//----------------------------------------------------------------------------------------------------------------------------------

#endif // MACROSIM_H
