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

/**\file DiffRayField_RayAiming.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef DIFFRAYFIELD_RAYAIMING_H
  #define DIFFRAYFIELD_RAYAIMING_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "DiffRayField.h"
#include "Detector.h"
//#include "Pupil.h"
#include <ctime>
#include "pugixml.hpp"

#define DIFFRAYFIELD_RAYAIMING_PATHTOPTX "macrosim_tracer_generated_rayGenerationDiffRayField_RayAiming.cu.ptx"

/* declare class */
/**
  *\class   rayFieldParams
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
class DiffRayField_RayAimingParams : public diffRayFieldParams
{
public:
	detParams oDetParams;	
    double3 initialTarget;
};

/* declare class */
/**
  *\class   DiffRayField_RayAiming
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
class DiffRayField_RayAiming : public DiffRayField
{
  protected:
	double3 oldPosition;
//	Pupil **pupilList;
	diffRayStruct* rayList;

	fieldError write2TextFile(char* filename, detParams &oDetParams);
	
	DiffRayField_RayAimingParams *rayParamsPtr;

	fieldError aimRay(diffRayStruct &ray, unsigned long long iX, unsigned long long iY);

  public:
    /* standard constructor */
    DiffRayField_RayAiming()
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, DIFFRAYFIELD_RAYAIMING_PATHTOPTX  );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator

		rayList = NULL;
		rayListLength=0;
		materialList=NULL;
		materialListLength=0;
		rayParamsPtr=new DiffRayField_RayAimingParams();
	}
    /* Konstruktor */
    DiffRayField_RayAiming(unsigned long long length)
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, DIFFRAYFIELD_RAYAIMING_PATHTOPTX  );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator
		rayList = (diffRayStruct*) malloc(length*sizeof(diffRayStruct));
		rayListLength = length;
		materialList=NULL;
		materialListLength=0;
		rayParamsPtr=new DiffRayField_RayAimingParams();
	}
	/* Destruktor */
	~DiffRayField_RayAiming()
	{
	  if ( materialList != NULL)
	  {
		  int i;
		  for (i=0;i<materialListLength;i++)
		  {
			  if (materialList[i]!=NULL)
			  {
				  delete materialList[i];
				  materialList[i]=NULL;
			  }
		  }
		  delete materialList;
		  materialList = NULL;
	  }
	  if ( rayList != NULL )
	  {
		delete rayList;
		rayList = NULL;
	  }
	  if (rayParamsPtr != NULL)
	  {
		  delete rayParamsPtr;
		  rayParamsPtr = NULL;
	  }

	}
	virtual long2 calcSubsetDim();

	virtual void setParamsPtr(DiffRayField_RayAimingParams *paramsPtr);
	virtual DiffRayField_RayAimingParams* getParamsPtr(void);
	fieldError setLambda(double lambda);

	virtual fieldError setRay(diffRayStruct ray, unsigned long long index);
	virtual diffRayStruct* getRay(unsigned long long index);
	virtual unsigned long long getRayListLength(void);
	virtual diffRayStruct* getRayList(void);
	virtual fieldError copyRayList(diffRayStruct *data, long long length);
	virtual fieldError copyRayListSubset(diffRayStruct *data, long2 launchOffset, long2 subsetDim);

    virtual fieldError createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);
	virtual fieldError createCPUSimInstance();
	virtual fieldError createLayoutInstance();

	virtual fieldError initCPUSubset();
    fieldError doSim(Group &oGroup, simAssParams &params, bool &simDone);

	virtual fieldError traceScene(Group &oGroup, bool RunOnCPU);
	virtual fieldError traceStep(Group &oGroup, bool RunOnCPU);

	virtual fieldError writeData2File(FILE *hFile_pos, rayDataOutParams outParams);

	virtual fieldError convert2RayData(Field** imagePtrPtr, detParams &oDetParams);
	virtual fieldError convert2Intensity(Field* imagePtr, detParams &oDetParams);
	virtual fieldError convert2ScalarField(Field* imagePtr, detParams &oDetParams);
	virtual fieldError convert2PhaseSpace(Field* imagePtr, detParams &oDetParams);
//	fieldError convert2VecField(Field* imagePtr, detParams &oDetParams);
	virtual fieldError processParseResults(FieldParseParamStruct &parseResults_Src, parseGlassResultStruct* parseResults_GlassPtr, DetectorParseParamStruct &parseResults_Det);
	virtual fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams);

};

#endif

