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

/**\file SimAssistant_Layout.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SIMASSISTANTLAYOUT_H
  #define SIMASSISTANTLAYOUT_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "Group.h"
#include "inputOutput.h"
#include "RayField.h"
#include "SimAssistant.h"
#include "MatlabInterface.h"
#include "MacroSimLib.h"

///* declare class */
///**
//  *\class   simAssLayoutParams 
//  *\ingroup SimAssistant
//  *\brief   
//  *
//  *         
//  *
//  *         \todo
//  *         \remarks           
//  *         \sa       NA
//  *         \date     04.01.2011
//  *         \author  Mauch
//  *
//  */
//class simAssLayoutParams : public simAssParams
//{
//public:
//
//};

/* declare class */
/**
  *\class   SimAssistantLayout
  *\ingroup SimAssistant
  *\brief   
  *
  *         
  *
  *         \todo
  *         \remarks           
  *         \sa       NA
  *         \date     04.01.2011
  *         \author  Mauch
  *
  */
class SimAssistantLayout : public SimAssistant
{
  protected:
	char* path_to_ptx;

//	simAssLayoutParams *paramsPtr;
//	MatlabInterface	oMatlabInterface;

	void (*callbackRayPlotData)(void* p2Object, double* rayPlotData, RayPlotDataParams *params); //!> function pointer to the callback function for the ray plot data

  public:
    /* standard constructor */
    SimAssistantLayout()
	{
		this->oFieldPtr=NULL;
		paramsPtr=NULL;
		path_to_ptx=(char*)malloc(512*sizeof(char));
//		this->oMatlabInterface=MatlabInterface();
	}
	/* Destruktor */
	~SimAssistantLayout()
	{
	  if ( this->oFieldPtr != NULL)
	  {
		  delete (this->oFieldPtr);
		  this->oFieldPtr=NULL;
	  }
	  if ( paramsPtr != NULL)
	  {
		  delete paramsPtr;
		  paramsPtr = NULL;
	  }
	  delete path_to_ptx;

	}

   	void setPathToPtx(char* path);
	const char* getPathToPtx(void);

	void setCallbackRayPlotData(void* p2CallbackObjectIn, void (*callbackRayPlotDataIn)(void* p2Object, double* rayPlotData, RayPlotDataParams *params))
		{
			this->p2CallbackObject=p2CallbackObjectIn;
			this->callbackRayPlotData=callbackRayPlotDataIn;
		};

	simAssError initSimulation( Group *oGroupPtr, Field *SourceListPtrPtr);
	virtual simAssError run(Group *oGroupPtr, RayField *SourceListPtrPtr, Detector **DetectorListPtrPtr);	
};

#endif

