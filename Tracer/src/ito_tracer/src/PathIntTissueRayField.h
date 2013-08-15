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

/**\file PathIntTissueRayField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef PATHINTTISSUERAYFIELD_H
  #define PATHINTTISSUERAYFIELD_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "RayField.h"
#include "wavefrontIn.h"
#include "inputOutput.h"
#include <ctime>
#include "pugixml.hpp"


#define PATHINTTISSUERAYFIELD_PATHTOPTX "ITO-MacroSim_generated_rayGenerationPathIntTissueRayField.cu.ptx"

/* declare class */
/**
  *\class   pathIntTissueRayFieldParams
  *\ingroup Field
  *\brief   params of a ray field
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
class pathIntTissueRayFieldParams : public rayFieldParams
{
public:
//	double lambda;
//	double3 root;
//	double3 tilt;
//	double power;
//	unsigned long width;
//	unsigned long height;
//	unsigned long widthLayout;
//	unsigned long heightLayout;
	double3 volumeWidth;
	double3 sourcePos;
	double meanFreePath;
	double anisotropy;
//	double nImmersed;
//	unsigned long long totalLaunch_width; //!> width of total rayfield
//	unsigned long long totalLaunch_height; //!> height of total rayfield
//	unsigned long long launchOffsetX; //!> 
//	unsigned long long launchOffsetY; //!>
//	unsigned long long GPUSubset_width;
//	unsigned long long GPUSubset_height;
};

/* declare class */
/**
  *\class   PathIntTissueRayField
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
class PathIntTissueRayField : public RayField
{
private:
	bool traceRay(rayStruct &ray);

  protected:
	//char path_to_ptx_rayGeneration[512];
	rayStruct* rayList;
	pathIntTissueRayFieldParams *rayParamsPtr;
//	fieldError write2MatFile(char* filename, detParams &oDetParams);
	fieldError write2TextFile(char* filename, detParams &oDetParams);
	
  public:
    /* standard constructor */
    PathIntTissueRayField()
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATHINTTISSUERAYFIELD_PATHTOPTX );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator
		rayList = NULL;
		rayListLength=0;
		materialList=NULL;
		materialListLength=0;
		rayParamsPtr=new pathIntTissueRayFieldParams();
	}
    /* Konstruktor */
    PathIntTissueRayField(unsigned long long length)
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATHINTTISSUERAYFIELD_PATHTOPTX );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator
		rayList = (rayStruct*) malloc(length*sizeof(rayStruct));
		rayListLength = length;
		materialList=NULL;//new Material*[1];
		materialListLength=0;
		rayParamsPtr=new pathIntTissueRayFieldParams();
	}
	/* Destruktor */
	~PathIntTissueRayField()
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
	
	void setParamsPtr(pathIntTissueRayFieldParams *paramsPtr);
	pathIntTissueRayFieldParams* getParamsPtr(void);

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

	fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec);
};

#endif

