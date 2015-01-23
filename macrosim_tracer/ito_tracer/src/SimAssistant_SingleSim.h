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

/**\file SimAssistant_SingleSim.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SIMASSISTANTSINGLESIM_H
  #define SIMASSISTANTSINGLESIM_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "Group.h"
#include "inputOutput.h"
#include "RayField.h"
#include "SimAssistant.h"

///* declare class */
///**
//  *\class   simAssSingleSimParams 
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
//class simAssSingleSimParams : public simAssParams
//{
//public:
//
//};

/* declare class */
/**
  *\class   SimAssistantSingleSim
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
class SimAssistantSingleSim : public SimAssistant
{
  protected:
	char* path_to_ptx;

	//simAssSingleSimParams *paramsPtr;


  public:
    /* standard constructor */
    SimAssistantSingleSim()
	{
		this->oFieldPtr=NULL;
		paramsPtr=NULL;
		path_to_ptx=(char*)malloc(512*sizeof(char));
	}
	/* Destruktor */
	~SimAssistantSingleSim()
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

//	virtual void setParamsPtr(simAssParams *paramsPtr);
//	virtual simAssSingleSimParams* getParamsPtr(void);

//	virtual simAssError initSimulation( Group *oGroupPtr, Field *SourceListPtrPtr);
	virtual simAssError run(Group *oGroupPtr, Field *SourceListPtrPtr, Detector **DetectorListPtrPtr);	
};

#endif

