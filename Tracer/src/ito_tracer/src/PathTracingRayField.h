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

/**\file PathTracingRayField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef PATHTRACINGRAYFIELD_H
  #define PATHTRACINGRAYFIELD_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "GeometricRayField.h"
#include "inputOutput.h"
#include <ctime>


#define PATHTRACINGRAYFIELD_PATHTOPTX "ITO-MacroSim_generated_rayGeneration_PathTracing.cu.ptx"

/* declare class */
/**
  *\class   PathTracingRayField
  *\ingroup Field
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
class PathTracingRayField : public GeometricRayField
{
  protected:
	 double3 oldPosition;
	//char path_to_ptx_rayGeneration[512];
	rayStruct_PathTracing* rayList;
//	rayFieldParams *rayParamsPtr; // see parent
	fieldError write2TextFile(char* filename, detParams &oDetParams);
	
  public:
    /* standard constructor */
    PathTracingRayField()
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATHTRACINGRAYFIELD_PATHTOPTX );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator
		rayList = NULL;
		rayListLength=0;
		materialList=NULL;
		materialListLength=0;
	}
    /* Konstruktor */
    PathTracingRayField(unsigned long long length)
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATHTRACINGRAYFIELD_PATHTOPTX );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator
		rayList = (rayStruct_PathTracing*) malloc(length*sizeof(rayStruct_PathTracing));
		rayListLength = length;
		materialList=NULL;
		materialListLength=0;
	}
	/* Destruktor */
	~PathTracingRayField()
	{
		if (materialList != NULL)
		{
			for (int i=0;i<materialListLength;i++)
			{
				if (materialList[i] != NULL)
				{
					delete materialList[i];
					materialList[i]=NULL;
				}
			}
			delete materialList;
			materialList=NULL;
		}
		if (rayList!=NULL)
		{
			delete rayList;
			rayList = NULL;
		}
	}
						
	virtual long2 calcSubsetDim(); // see parent

	virtual fieldError setRay(rayStruct_PathTracing ray, unsigned long long index);
	virtual rayStruct_PathTracing* getRay(unsigned long long index);
	virtual rayStruct_PathTracing* getRayList(void);
	virtual void setRayList(rayStruct_PathTracing* rayStruct_PathTracingPtr);
	virtual fieldError copyRayList(rayStruct_PathTracing *data, long long length);
	virtual fieldError copyRayListSubset(rayStruct_PathTracing *data, long2 launchOffset, long2 subsetDim);

	virtual fieldError createCPUSimInstance(); // see parent

	virtual fieldError initCPUSubset();

    virtual fieldError createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);

	virtual fieldError traceScene(Group &oGroup, bool RunOnCPU, RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);
	virtual fieldError traceStep(Group &oGroup, bool RunOnCPU, RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);

	//fieldError writeData2File(FILE *hFile_pos, rayDataOutParams outParams);

	virtual fieldError convert2Intensity(Field* imagePtr, detParams &oDetParams);
	virtual fieldError convert2ScalarField(Field* imagePtr, detParams &oDetParams);
	virtual fieldError convert2VecField(Field* imagePtr, detParams &oDetParams);
	
	virtual fieldError convert2RayData(Field** imagePtr, detParams &oDetParams);
	virtual fieldError processParseResults(FieldParseParamStruct &parseResults_Src, parseGlassResultStruct* parseResults_GlassPtr);
};

#endif

