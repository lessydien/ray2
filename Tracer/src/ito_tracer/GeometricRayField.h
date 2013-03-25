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

/**\file GeometricRayField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef GEOMRAYFIELD_H
  #define GEOMRAYFIELD_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "RayField.h"
#include "wavefrontIn.h"
#include "inputOutput.h"
#include <ctime>
#include "pugixml.hpp"


#define GEOMRAYFIELD_PATHTOPTX "ITO-MacroSim_generated_rayGeneration.cu.ptx"

/* declare class */
/**
  *\class   GeometricRayField
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
class GeometricRayField : public RayField
{
  protected:
	//char path_to_ptx_rayGeneration[512];
	rayStruct* rayList;
	rayFieldParams *rayParamsPtr;
//	fieldError write2MatFile(char* filename, detParams &oDetParams);
	fieldError write2TextFile(char* filename, detParams &oDetParams);
	
  public:
    /* standard constructor */
    GeometricRayField()
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, GEOMRAYFIELD_PATHTOPTX );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator
		rayList = NULL;
		rayListLength=0;
		materialList=NULL;
		materialListLength=0;
		rayParamsPtr=new rayFieldParams();
	}
    /* Konstruktor */
    GeometricRayField(unsigned long long length)
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, GEOMRAYFIELD_PATHTOPTX );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator
		rayList = (rayStruct*) malloc(length*sizeof(rayStruct));
		rayListLength = length;
		materialList=NULL;//new Material*[1];
		materialListLength=0;
		rayParamsPtr=new rayFieldParams();
	}
	/* Destruktor */
	~GeometricRayField()
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
		if (rayParamsPtr != NULL)
		{
			delete rayParamsPtr;
			rayParamsPtr=NULL;
		}
	}
					
	long2 calcSubsetDim();

   	void setPathToPtx(char* path);
	const char* getPathToPtx(void);
	
	void setParamsPtr(rayFieldParams *paramsPtr);
	rayFieldParams* getParamsPtr(void);

	fieldError setRay(rayStruct ray, unsigned long long index);
	rayStruct* getRay(unsigned long long index);
	fieldError setLambda(double lambda);
	unsigned long long getRayListLength(void);
	rayStruct* getRayList(void);
	void setRayList(rayStruct* rayStructPtr);
	fieldError copyRayList(rayStruct *data, long long length);
	fieldError copyRayListSubset(rayStruct *data, long2 launchOffset, long2 subsetDim);

//	void createCPUSimInstance(unsigned long long nWidth,unsigned long long nHeight,double distance, double3 rayDirection,double3 firstRayPosition, double flux, double lambda);
	virtual fieldError createCPUSimInstance();
	virtual fieldError createLayoutInstance();

	fieldError initGPUSubset(RTcontext &context, RTbuffer &seed_buffer_obj);
	fieldError initCPUSubset();

    fieldError createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);
	fieldError createOptixInstance(RTcontext* context, unsigned long long width, unsigned long long height, double3 start, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double RadiusSourceReference);

	fieldError traceScene(Group &oGroup);
	fieldError traceScene(Group &oGroup, bool RunOnCPU);
	fieldError traceStep(Group &oGroup, bool RunOnCPU);

	fieldError doSim(Group &oGroup, simAssParams &params, bool &simDone);

	//fieldError writeData2File(FILE *hFile_pos, rayDataOutParams outParams);

	fieldError convert2Intensity(Field* imagePtr, detParams &oDetParams);
	fieldError convert2ScalarField(Field* imagePtr, detParams &oDetParams);
	fieldError convert2VecField(Field* imagePtr, detParams &oDetParams);
	fieldError convert2PhaseSpace(Field* imagePtr, detParams &oDetParams);
	
	fieldError convert2RayData(Field** imagePtr, detParams &oDetParams);
	fieldError processParseResults(FieldParseParamStruct &parseResults_Src, parseGlassResultStruct* parseResults_GlassPtr);
	fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec);
};

#endif

