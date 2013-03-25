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

/**\file Field.h
* \brief 
* 
*           
* \author Mauch
*/

/**
 *\defgroup Field
 */

#ifndef FIELD_H
  #define FIELD_H

#include <optix.h>
#include "stdlib.h"
#include <stdio.h>
//#include "complex.h"
#include "my_vector_types.h"
//#include "optix_math.h"
#include "DetectorParams.h"
#include "Group.h"
#include "Material.h"
#include "Geometry_intersect.h"
#include "pugixml.hpp"
#include "FieldParams.h"
#include <iostream>

//typedef enum 
//{
//  FIELD_NO_ERR,
//  FIELD_ERR,
//  FIELD_INDEXOUTOFRANGE_ERR
//} fieldError;
//
//typedef enum
//{
//	metric_m,
//	metric_mm,
//	metric_mu,
//	metric_nm,
//	metric_au // arbitray units
//} metric_unit;
//
//typedef struct
//{
//	metric_unit x;
//	metric_unit y;
//	metric_unit z;
//} axesUnits;

//typedef struct
//{
//  geometry_type type;
//  unsigned long long width; //!> in grid rect: number of rays along x-axis; in grid rad: number of rays along radial direction
//  unsigned long long height; //!> in grid rect: number of rays along y-axis; in grid rad: number of rays along angular direction
//  unsigned long long widthLayout; //!> analog to width for LayoutMode
//  unsigned long long heightLayout; //!> analog to height for LayoutMode
//  double power;
//  rayDirDistrType rayDirDistr;
//  rayPosDistrType rayPosDistr;
//  double3 rayDirection;
//  double lambda;
//  double2 apertureHalfWidth1;
//  double2 apertureHalfWidth2;
////  double rotNormal1;
//  MaterialParseParamStruct materialParams;
//  double3 root;
//  double3 normal;
//  double2 alphaMax; //!> maximum angle relative to source normal of the cone into which raydirections are uniformly distributed along x and y
//  double2 alphaMin; //!> minimum angle relative to source normal of the cone into which raydirections are uniformly distributed along x and y
//  double coherence;
//  ulong2 nrRayDirections; //!> nr of rays that are shot in random directions from each point source in case of diiferential ray source
//  double epsilon; //!> short distance that the rays are moved from the caustic of each point source in case of differential ray sources
//  double3 tilt;
//  int importanceObjNr;
//  double2 importanceConeAlphaMax;
//  double2 importanceConeAlphaMin;
//  double3 importanceAreaTilt; 
//  double3 importanceAreaRoot;
//  double2 importanceAreaHalfWidth;
//  ApertureType importanceAreaApertureType;
//  bool importanceArea;
////	int importanceObjNr;
//
//} FieldParseParamStruct;

///* declare class */
///**
//  *\class   fieldParams
//  *\ingroup Field
//  *\brief   
//  *
//  *         
//  *
//  *         \todo
//  *         \remarks           
//  *         \sa       NA
//  *         \date     04.01.2011
//  *         \author  Mauch
//  *
//  */
//class fieldParams
//{
//public:
//	long3 nrPixels; // number of pixels in each direction
//	double4x4 MTransform; 
//	double3 scale; // physical size of voxel in each dimension
//	axesUnits units;
//	double lambda;
//	metric_unit unitLambda;
//
//	/* standard constructor */
//	//fieldParams()
//	//{
//	//	this->nrPixels=make_long3(0,0,0);
//	//	this->MTransform=make_double4x4(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
//	//	this->scale=make_double3(0,0,0);
//	//	this->unitLambda=metric_au;
//	//	axesUnits l_axesUnits;
//	//	l_axesUnits.x=metric_au;
//	//	l_axesUnits.y=metric_au;
//	//	l_axesUnits.z=metric_au;
//	//	this->units=l_axesUnits;
//	//}
//	/* destructor */
//	//~fieldParams()
//	//{
//	//}
//
//};

/* declare class */
/**
  *\class   Field
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
class Field
{
  protected:

	void* p2ProgCallbackObject; //!> pointer to the object that holds the callback function
	void (*callbackProgress)(void* p2Object, int progressValue); //!> function pointer to the callback function

	virtual fieldError write2TextFile(char* filename, detParams &oDetParams);
	virtual fieldError write2MatFile(char* filename, detParams &oDetParams);
	fieldError Field::convertFieldParams2ItomFieldParams(ItomFieldParams* paramsOut);

	unsigned long long GPU_SUBSET_WIDTH_MAX;
	unsigned long long GPU_SUBSET_HEIGHT_MAX;

	int numCPU;

  public:
    /* standard constructor */
    Field()
	{
		numCPU=1;
	}
	/* Destruktor */
	virtual ~Field()
	{

	}

	void setSubsetWidthMax(unsigned long long in) {GPU_SUBSET_WIDTH_MAX=in;};
	unsigned long long getSubsetWidthMax() {return GPU_SUBSET_WIDTH_MAX;};
	void setSubsetHeightMax(unsigned long long in) {GPU_SUBSET_HEIGHT_MAX=in;};
	unsigned long long getSubsetHeightMax() {return GPU_SUBSET_HEIGHT_MAX;};
	void setNumCPU(int in){numCPU=in;};
	int getNumCPU() {return numCPU;};

	void setProgressCallback(void* p2CallbackObjectIn, void (*callbackProgressIn)(void* p2Object, int progressValue));

	virtual fieldError write2File(char* filename, detParams &oDetParams);

	virtual fieldParams* getParamsPtr(void);

	virtual fieldError processParseResults(FieldParseParamStruct &parseResults_Src);
	virtual fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec);

	//virtual fieldError writeData2File(FILE *hFile_pos, rayDataOutParams outParams);
	virtual fieldError convert2Intensity(Field *imagePtr, detParams &oDetParams);
	virtual fieldError convert2PhaseSpace(Field* imagePtr, detParams &oDetParams);
	virtual fieldError convert2ScalarField(Field* imagePtr, detParams &oDetParams);
	virtual fieldError convert2VecField(Field* imagePtr, detParams &oDetParams);
	virtual fieldError convert2RayData(Field **imagePtrPtr, detParams &oDetParams);
	virtual fieldError convert2ItomObject(void** dataPtrPtr, ItomFieldParams* paramsOut);
	virtual fieldError traceScene(Group &oGroup, bool RunOnCPU);
	virtual fieldError traceStep(Group &oGroup, bool RunOnCPU);

	virtual fieldError initGPUSubset(RTcontext &context);
	virtual fieldError initCPUSubset();
	virtual fieldError initSimulation(Group &oGroup, simAssParams &params);
	virtual fieldError initLayout(Group &oGroup, simAssParams &params);

	virtual fieldError createCPUSimInstance();
	virtual fieldError createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);
	virtual fieldError  createLayoutInstance();

	virtual fieldError  doSim(Group &oGroup, simAssParams &params, bool &simDone);
};

#endif

