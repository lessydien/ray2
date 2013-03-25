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

/**\file Detector.h
* \brief 
* 
*           
* \author Mauch
*/

/**
 *\defgroup Detector
 */

#ifndef DETECTOR_H
  #define DETECTOR_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
//#include "Group.h"
#include "inputOutput.h"
#include "DetectorParams.h"
#include "Field.h"
#include "pugixml.hpp"
//#include "Geometry.h"

typedef enum 
{
  DET_NO_ERROR,
  DET_ERROR
} detError;

/* declare class */
/**
  *\class   Detector
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
class Detector
{
  protected:
	char* path_to_ptx;

	detParams *detParamsPtr;
	Field* oFieldPtr;


  public:
    /* standard constructor */
    Detector()
	{
		detParamsPtr=NULL;
		path_to_ptx=(char*)malloc(512*sizeof(char));
	}
	/* Destruktor */
	~Detector()
	{
	  if ( detParamsPtr != NULL)
	  {
		  detParamsPtr = NULL;
	  }
	  delete path_to_ptx;
	}

   	void setPathToPtx(char* path);
	const char* getPathToPtx(void);

	virtual void setDetParamsPtr(detParams *paramsPtr);
	virtual detParams* getDetParamsPtr(void);

	virtual detError processParseResults(DetectorParseParamStruct &parseResults_Det);
	virtual detError parseXml(pugi::xml_node &det, vector<Detector*> &detVec);

	virtual detError detect2TextFile(FILE* hfile, Field* rayFieldPtr);	
	virtual detError detect(Field* rayFieldPtr, Field **imagePtrPtr);
};

#endif

