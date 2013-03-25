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

/**\file SimAssistant_ParamSweep.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SIMASSISTANTPARAMSWEEP_H
  #define SIMASSISTANTPARAMSWEEP_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "Group.h"
#include "inputOutput.h"
#include "RayField.h"
#include "Geometry.h"
#include "RayField.h"
#include "Detector.h"
#include "SimAssistant.h"

typedef struct
{
	Geometry_Params *geometryParams;
	unsigned int geomObjectIndex; //!> indices of the objects whose parameters are swept
} Geometry_Sweep_Params;

/* declare class */
/**
  *\class   simAssParamSweepParams 
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
class simAssParamSweepParams : public simAssParams
{
public:
	Geometry_Sweep_Params **geometrySweepParamsList;
	unsigned long long geomParamsSweepLength;
	unsigned long *geomParamsIndividSweepLengths; //!> length of sweep for each parameter
	unsigned int geomParamsTotalSweepDim; //!> number of parameters that are to be swept
	rayFieldParams **srcParamsList;
	unsigned long long srcParamsSweepLength;
	int srcIndex;
	detParams **detParamsList;
	unsigned long long detParamsSweepLength;
	int detIndex;
};

/* declare class */
/**
  *\class   SimAssistantParamSweep
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
class SimAssistantParamSweep : public SimAssistant
{
  protected:
	char* path_to_ptx;

	simAssParamSweepParams *paramsPtr;
	Field** FieldPtrList;
	unsigned long long fieldPtrListLength;

	simAssError updateSimulation( Group *oGroupPtr, RayField *SourceListPtrPtr, simAssParams *paramsPtr);


  public:
    /* standard constructor */
    SimAssistantParamSweep()
	{
		paramsPtr=NULL;
		FieldPtrList=NULL;
		fieldPtrListLength=0;
		path_to_ptx=(char*)malloc(512*sizeof(char));
	}
    /* additional constructor */
    SimAssistantParamSweep(simAssParamSweepParams *paramsPtrIn)
	{
		paramsPtr=paramsPtrIn;
		fieldPtrListLength=(paramsPtr->detParamsSweepLength+1)*(paramsPtr->srcParamsSweepLength+1)*(paramsPtr->geomParamsSweepLength+1);// if Listlength equals zero for one object type, we still need to do one simulation for the original parameter set...
		FieldPtrList=new Field*[fieldPtrListLength];
		for (unsigned long long j=0;j<fieldPtrListLength;j++)
		{
			FieldPtrList[j]=NULL;
		}
		delete path_to_ptx;
	}

	/* Destruktor */
	~SimAssistantParamSweep()
	{
	  if ( paramsPtr != NULL)
	  {
		  if (paramsPtr->detParamsList != NULL)
		  {
			  for (int j=0;j<paramsPtr->detParamsSweepLength;j++)
			  {
				  delete paramsPtr->detParamsList[j];
				  paramsPtr->detParamsList[j]=NULL;
			  }
			  for (int j=0;j<paramsPtr->srcParamsSweepLength;j++)
			  {
				  delete paramsPtr->srcParamsList[j];
				  paramsPtr->srcParamsList[j]=NULL;
			  }
			  for (int j=0;j<paramsPtr->geomParamsSweepLength;j++)
			  {
				  delete paramsPtr->geometrySweepParamsList[j];
				  paramsPtr->geometrySweepParamsList[j]=NULL;
			  }
			  delete paramsPtr;
			  paramsPtr = NULL;
		  }
	  }
	  if (FieldPtrList != NULL)
	  {
		  for (int j=0;j<fieldPtrListLength;j++)
		  {
			  delete FieldPtrList[fieldPtrListLength-1-j];
			  FieldPtrList[fieldPtrListLength-1-j]=NULL;
		  }
		  delete FieldPtrList;
		  FieldPtrList=NULL;	
	  }
	}

   	void setPathToPtx(char* path);
	const char* getPathToPtx(void);

	virtual void setParamsPtr(simAssParams *paramsPtr);
	virtual simAssParamSweepParams* getParamsPtr(void);

	virtual simAssError initSimulation( Group *oGroupPtr, Field *SourceListPtrPtr);
	virtual simAssError run(Group *oGroupPtr, RayField *SourceListPtrPtr, Detector **DetectorListPtrPtr);	
};

#endif

