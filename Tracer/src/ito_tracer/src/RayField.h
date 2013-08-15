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

/**\file RayField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef RAYFIELD_H
  #define RAYFIELD_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "Group.h"
#include "inputOutput.h"
#include "MaterialRefracting.h"
#include "Field.h"
#include "MacroSimLib.h"


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
class rayFieldParams : public fieldParams
{
public:
	double3x3 Mrot;
	double3 tilt;
	double3 translation;
	double3 rayPosStart;
	double3 rayPosEnd;
	double3 rayDirection;
	ulong2 nrRayDirections; //!> number of ray directions in x- and y-direction (or in r- and theta- directions in case of radial distributions)
//	long2 nrRayPositions; //!> number of ray positions in x- and y-direction (or in r- and theta- directions in case of radial distributions)
	double flux;
//	double lambda; // defined in base class
	unsigned long long width; //!> number of ray positions in x-direction (or in r- directions in case of radial distributions) in current launch
	unsigned long long height; //!> number of ray positions in y-direction (or in theta- directions in case of radial distributions) in current launch
	unsigned long long widthLayout; //!> analog to width in LayoutMode
	unsigned long long heightLayout; //!> analog to height in LayoutMode
	unsigned long long totalLaunch_width; //!> width of total rayfield
	unsigned long long totalLaunch_height; //!> height of total rayfield
	unsigned long long launchOffsetX; //!> 
	unsigned long long launchOffsetY; //!>
	unsigned long long GPUSubset_width;
	unsigned long long GPUSubset_height;
	unsigned long long layout_width; //!> total width of rayfield for layout mode
	unsigned long long layout_height; //!> total height of rayfield for layout mode
	rayPosDistrType posDistrType;
	rayDirDistrType dirDistrType;
	double nImmersed;
	double2 alphaMax; //!> maximum angle relative to source normal of the cone into which raydirections are uniformly distributed along x and y
	double2 alphaMin; //!> minimum angle relative to source normal of the cone into which raydirections are uniformly distributed along x and y
//	double4x4 MTransf; //!> transformation matrix of source // defined in base class
	double coherence; //!> complex coherence parameter
	bool importanceArea; //!> flag signaling the existence of an importanceArea for the source
	double3 importanceAreaTilt; 
	double3 importanceAreaRoot;
	double2 importanceAreaHalfWidth;
	ApertureType importanceAreaApertureType;
//	ImpAreaType importanceAreaType;
//	int importanceObjNr;
};

/* declare class */
/**
  *\class   RayField
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
class RayField : public Field
{
  private:
	fieldError createOptiXContext();

  protected:
	RTcontext context; //!> this is where the instances of the OptiX simulation will be stored
	RTbuffer output_buffer_obj; //!> this is where the buffers for the OptiX simulation will be stored
	RTbuffer   seed_buffer_obj; //!> this is where the buffers for the OptiX simulation will be stored

	unsigned long long tracedRayNr;
	char path_to_ptx_rayGeneration[512];
//	rayStructBase* rayList;
	unsigned long long rayListLength;
	uint32_t x[5];
//	rayFieldParams *rayParamsPtr;

    Material** materialList;
	int materialListLength;

	bool update;
	unsigned long long subsetCounter;

	fieldError calcImmersion();
	RTprogram  ray_gen_program;


  public:
    /* standard constructor */
    RayField()
	{
//		rayList = NULL;
		rayListLength=0;
		subsetCounter=0;
		update=true;
		materialList=NULL;
		materialListLength=0;
	}
    /* Konstruktor */
    RayField(unsigned long long length)
	{
//	  rayList = (rayStructBase*) malloc(length*sizeof(rayStructBase));
	  rayListLength = length;
	  subsetCounter=0;
	  update=true;
	  materialList=NULL;
	  materialListLength=0;
	}
	/* Destruktor */
	~RayField()
	{
	  if ( materialList != NULL)
	  {
		  int i;
		  for (i=0;i<materialListLength;i++)
		  {
			  if (materialList[i]!=NULL)
			  {
				  delete materialList[i];
			  }
		  }
		  materialList = NULL;
	  }
	 // if ( rayList != NULL )
	 // {
		//delete rayList;
		//rayList = NULL;
	 // }
	}

   	void setPathToPtx(char* path);
	const char* getPathToPtx(void);
	unsigned long long getRayListLength(void);

	virtual long2 calcSubsetDim();
	virtual void setParamsPtr(rayFieldParams *paramsPtr);
	virtual rayFieldParams* getParamsPtr(void);

	virtual fieldError setRay(rayStructBase ray, unsigned long long index);
	virtual rayStructBase* getRay(unsigned long long index);
	virtual rayStructBase* getRayList(void);
	Material* getMaterial(int index);
	fieldError setMaterial(Material *oMaterialPtr, int index);
	fieldError setMaterialListLength(int length);
	int getMaterialListLength(void);



	virtual fieldError copyRayList(rayStruct *data, long long length);
	virtual fieldError copyRayListSubset(rayStruct *data, long2 launchOffset, long2 subsetDim);
	virtual fieldError setLambda(double lambda);

//	virtual void createCPUSimInstance(unsigned long long nWidth,unsigned long long nHeight,double distance, double3 rayDirection,double3 firstRayPosition, double flux, double lambda);
	virtual fieldError createCPUSimInstance();
	virtual void createCPUSimInstance(unsigned long long launch_width, unsigned long long launch_height, double3 start, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double lambda);
	virtual fieldError createLayoutInstance();

	virtual fieldError parseXml(pugi::xml_node &det, vector<Field*> &fieldVec);

	virtual fieldError initGPUSubset(RTcontext &context, RTbuffer &seed_buffer_obj);
	virtual fieldError initCPUSubset();

    virtual fieldError createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);
	virtual fieldError createOptixInstance(RTcontext* context, unsigned long long width, unsigned long long height, double3 start, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double RadiusSourceReference);

	virtual fieldError initGPUSubset(RTcontext &context);
	virtual fieldError initSimulation(Group &oGroup, simAssParams &params);
	virtual fieldError initLayout(Group &oGroup, simAssParams &params);
		
	virtual fieldError traceScene(Group &oGroup);
	virtual fieldError traceScene(Group &oGroup, bool RunOnCPU);
	virtual fieldError traceStep(Group &oGroup, bool RunOnCPU);

	virtual fieldError doSim(Group &oGroup, simAssParams &params, bool &simDone);
//	virtual fieldError writeData2File(FILE *hFile_pos, rayDataOutParams outParams);
};

#endif

