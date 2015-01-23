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

/**\file Detector_Field.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef DETECTOR_FIELD_H
  #define DETECTOR_FIELD_H

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
  *\class   detFieldParams
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
class detFieldParams: public detParams
{
public:

};

/* declare class */
/**
  *\class   DetectorField
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
class DetectorField: public Detector
{
  protected:
	char* path_to_ptx;

	detFieldParams *detParamsPtr;


  public:
    /* standard constructor */
    DetectorField()
	{
		detParamsPtr=new detFieldParams();
		path_to_ptx=(char*)malloc(512*sizeof(char));
	}
	/* Destruktor */
	~DetectorField()
	{
	  if ( detParamsPtr != NULL)
	  {
		  detParamsPtr = NULL;
	  }
	  delete path_to_ptx;
	}

   	void setPathToPtx(char* path);
	const char* getPathToPtx(void);

	void setDetParamsPtr(detFieldParams *paramsPtr);
	detFieldParams* getDetParamsPtr(void);

	detError detect2TextFile(FILE* hfile, RayField* rayFieldPtr);
	detError detect(Field* rayFieldPtr, Field **imagePtrPtr);
	detError processParseResults(DetectorParseParamStruct &parseResults_Det);
	detError parseXml(pugi::xml_node &det, vector<Detector*> &detVec);
	
};

#endif

