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

/**\file SimAssistant.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SIMASSISTANT_H
  #define SIMASSISTANT_H

//#include <optix.h>
#include "stdlib.h"
#include "myUtil.h"
#include <iostream>
#include "Group.h"
#include "rayData.h"
#include "Detector.h"
#include "RayField.h"
#include "MacroSimLib.h"

typedef enum 
{
  SIMASS_NO_ERROR,
  SIMASS_ERROR
} simAssError;

/* declare class */
/**
  *\class   SimAssistant
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
class SimAssistant
{
  //private:
	//simAssError createOptiXContext();
  protected:
	char* path_to_ptx;
	simAssParams *paramsPtr;
	//RTcontext context; //!> this is where the instances of the OptiX simulation will be stored
	//RTbuffer output_buffer_obj; //!> this is where the buffers for the OptiX simulation will be stored
	//RTbuffer   seed_buffer_obj; //!> this is where the buffers for the OptiX simulation will be stored
	Field* oFieldPtr; //!> this is where the result of the simulation will be stored
	void* p2CallbackObject; //!> pointer to the object that holds the callback function
	void (*callbackProgress)(void* p2Object, int progressValue); //!> function pointer to the callback function

	//virtual simAssError initSimulationBaseClass( Group *oGroupPtr, Field *SourceListPtrPtr, simAssParams *paramsPtr);
	//virtual simAssError doSim(Group &oGroup, Field *SourceListPtrPtr, Detector **DetectorListPtrPtr, Field **ResultFieldPtrPtr, bool RunOnCPU, RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);
	virtual simAssError doSim(Group &oGroup, Field *SourceListPtrPtr, Detector **DetectorListPtrPtr, Field **ResultFieldPtrPtr, bool RunOnCPU);

  public:
    /* standard constructor */
    SimAssistant()
	{
		paramsPtr=NULL;
		path_to_ptx=(char*)malloc(512*sizeof(char));
	}
	/* Destruktor */
	virtual ~SimAssistant()
	{
	  if ( paramsPtr != NULL)
	  {
		  delete paramsPtr;
		  paramsPtr = NULL;
	  }
	  if (oFieldPtr != NULL )
	  {
		  delete oFieldPtr;
		  oFieldPtr = NULL;
	  }
	  delete path_to_ptx;
	}

   	void setPathToPtx(char* path);
	const char* getPathToPtx(void);

	void setCallback(void* p2CallbackObject, void (*callbackProgress)(void* p2Object, int progressValue));
	Field* getResultFieldPtr(void) {return oFieldPtr;};

	virtual void setParamsPtr(simAssParams *paramsPtr);
	virtual simAssParams* getParamsPtr(void);
	virtual simAssError initSimulation( Group *oGroupPtr, Field *SourceListPtrPtr);
	virtual simAssError run(Group *oGroupPtr, Field *SourceListPtrPtr, Detector **DetectorListPtrPtr);
};

#endif

