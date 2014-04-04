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

/**\file GeometricRenderField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef GEOMENDERFIELD_H
  #define GEOMRENDERFIELD_H

#include <optix.h>
#include "..\rayData.h"
#include "stdlib.h"
#include "..\RayField.h"
#include "..\inputOutput.h"
#include <ctime>
#include "..\pugixml.hpp"


#define GEOMRENDERFIELD_PATHTOPTX "ITO-MacroSim_generated_render.cu.ptx"

/* declare class */
/**
  *\class   renderFieldParams
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
class renderFieldParams : public fieldParams
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
	unsigned long long width; //!> number pixel in x-direction (or in r- directions in case of radial distributions) in current launch
	unsigned long long height; //!> number pixel in y-direction (or in theta- directions in case of radial distributions) in current launch
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
  *\class   GeometricRenderField
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
class GeometricRenderField : public Field
{
  protected:
    double* Iptr; //!> field for Intensity.

	char path_to_ptx_rayGeneration[512];
	renderFieldParams *renderFieldParamsPtr;
//	fieldError write2MatFile(char* filename, detParams &oDetParams);
	fieldError write2TextFile(char* filename, detParams &oDetParams);

    geomRenderRayStruct createRay(unsigned long long jx, unsigned long long jy, unsigned long long jRay);

    Material** materialList;
	int materialListLength;

    bool update;

    uint32_t x[5]; // seed for randomn generator

    unsigned long long subsetCounter;
    unsigned long long tracedRayNr;

	RTcontext context; //!> this is where the instances of the OptiX simulation will be stored
	RTbuffer output_buffer_obj; //!> this is where the buffers for the OptiX simulation will be stored
	RTbuffer   seed_buffer_obj; //!> this is where the buffers for the OptiX simulation will be stored

    RTprogram  ray_gen_program;
	
  public:
    /* standard constructor */
    GeometricRenderField()
	{
        Iptr=NULL;
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, GEOMRENDERFIELD_PATHTOPTX );		
		materialList=NULL;
		materialListLength=0;
		renderFieldParamsPtr=new renderFieldParams();
        // init seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x);

	}
    /* Konstruktor */
    GeometricRenderField(unsigned long long length)
	{
        Iptr=NULL;
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, GEOMRENDERFIELD_PATHTOPTX );		
		// init random seed
		materialList=NULL;//new Material*[1];
		materialListLength=0;
		renderFieldParamsPtr=new renderFieldParams();
        // init seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x);
	}
	/* Destruktor */
	~GeometricRenderField()
	{
		if (this->Iptr != NULL)
		{
			delete this->Iptr;
			this->Iptr=NULL;
		}
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
		if (renderFieldParamsPtr != NULL)
		{
			delete renderFieldParamsPtr;
			renderFieldParamsPtr=NULL;
		}
	}
					
	long2 calcSubsetDim();

   	void setPathToPtx(char* path);
	const char* getPathToPtx(void);

	Material* getMaterial(int index);
	fieldError setMaterial(Material *oMaterialPtr, int index);
	fieldError setMaterialListLength(int length);
	int getMaterialListLength(void);
	
	void setParamsPtr(renderFieldParams *paramsPtr);
	renderFieldParams* getParamsPtr(void);

	fieldError setRay(rayStruct ray, unsigned long long index);
	rayStruct* getRay(unsigned long long index);
	fieldError setLambda(double lambda);

//	void createCPUSimInstance(unsigned long long nWidth,unsigned long long nHeight,double distance, double3 rayDirection,double3 firstRayPosition, double flux, double lambda);
	virtual fieldError createCPUSimInstance();

    fieldError initSimulation(Group &oGroup, simAssParams &params);
    fieldError createOptiXContext();

	fieldError initGPUSubset(RTcontext &context, RTbuffer &seed_buffer_obj);

    fieldError createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);
	fieldError createOptixInstance(RTcontext* context, unsigned long long width, unsigned long long height, double3 start, double3 end, double* xGrad, int size_xGrad, double* yGrad, int size_yGrad, double RadiusSourceReference);

	fieldError traceScene(Group &oGroup, bool RunOnCPU);
	fieldError traceStep(Group &oGroup, bool RunOnCPU);

	fieldError doSim(Group &oGroup, simAssParams &params, bool &simDone);

    void setSimMode(SimMode &simMode);
	
	fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams);

    fieldError convert2Intensity(Field *imagePtr, detParams &oDetParams);
};

#endif

