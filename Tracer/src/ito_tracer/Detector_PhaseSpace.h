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

/**\file Detector_PhaseSpace.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef DETECTOR_PHASESPACE_H
  #define DETECTOR_PHASESPACE_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "Group.h"
#include "inputOutput.h"
#include "RayField.h"
#include "Detector.h"
#include "pugixml.hpp"

/* declare class */
/**
  *\class   detPhaseSpaceParams
  *\ingroup Detector
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
class detPhaseSpaceParams: public detParams
{
public:
	ulong2 detPixel_PhaseSpace;
	double2 dirHalfWidth;


};

/* declare class */
/**
  *\class   DetectorPhaseSpace
  *\ingroup Detector
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
class DetectorPhaseSpace: public Detector
{
  protected:
	char* path_to_ptx;

	detPhaseSpaceParams *detParamsPtr;


  public:
    /* standard constructor */
    DetectorPhaseSpace()
	{
		detParamsPtr=new detPhaseSpaceParams();
		path_to_ptx=(char*)malloc(512*sizeof(char));
	}
	/* Destruktor */
	~DetectorPhaseSpace()
	{
	  if ( detParamsPtr != NULL)
	  {
		  detParamsPtr = NULL;
	  }
	  delete path_to_ptx;
	}

   	void setPathToPtx(char* path);
	const char* getPathToPtx(void);

	void setDetParamsPtr(detPhaseSpaceParams *paramsPtr);
	detPhaseSpaceParams* getDetParamsPtr(void);

	detError detect2TextFile(FILE* hfile, RayField* rayFieldPtr);
	detError detect(Field* rayFieldPtr, Field **imagePtrPtr);
	detError processParseResults(DetectorParseParamStruct &parseResults_Det);
	detError parseXml(pugi::xml_node &det, vector<Detector*> &detVec);
	
};

#endif

