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

/**\file Geometry.h
* \brief 
* 
*           
* \author Mauch
*/

/**
 *\defgroup Geometry
 */

#ifndef GEOMETRY_H
  #define GEOMETRY_H

#include <optix.h>
#include "Material.h"
#include "DetectorParams.h"
#include "my_vector_functions.h" // from cuda toolkit
#include "rayData.h"
#include "stdio.h"
#include "sampleConfig.h"
#include "Geometry_intersect.h"
#include "pugixml.hpp"
#include <vector>

using namespace std;

typedef enum 
{
  GEOM_NO_ERR,
  GEOM_ERR,
  GEOM_LISTCREATION_ERR,
  GEOM_NOMATERIAL_ERR,
  GEOM_ADDMAT_UNKNOWNMAT_ERR,
  GEOM_GBINCONSISTENTINTERSECTIONS_ERR,
} geometryError;

typedef struct
{
  geometry_type type;
  double3 root;
  double3 normal;
  double3 tilt;
  double thickness;
  ApertureType aperture;
  double2 apertureHalfWidth1;
  double2 apertureHalfWidth2;
  double2 obscurationHalfWidth;
  char comment[GEOM_CMT_LENGTH];
  double2  radius1;
  double2  radius2;
//  double rotNormal1;
//  double rotNormal2;
  double  diameter;
  double  conic1;
  double  conic2;
  double params[8];
  double asphereParams[MAX_NR_MATPARAMS];
  double3 cosNormAxis;
  double cosNormAmpl;
  double cosNormPeriod;
  double iterationAccuracy;
  MaterialParseParamStruct materialParams;
  DetectorParseParamStruct detectorParams;
} GeometryParseParamStruct;

/* declare class */
/**
  *\class   Geometry_Params
  *\ingroup Geometry
  *\brief   base class of the params of all geometries
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
class Geometry_Params
{
public:
    double2 apertureRadius;
//    double rotNormal; // rotation of geometry around its normal
    ApertureType apertureType;
	int geometryID;
	double3 tilt;
	/* standard constructor */
	Geometry_Params()
	{
	}
	/* destructor */
	virtual ~Geometry_Params()
	{
	}
};

/* declare class */
/**
  *\class   Geometry
  *\ingroup Geometry
  *\brief   base class of all geometries
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
class Geometry
{
  protected:
    Material** materialList;
	int materialListLength;
	geometryError intersect();
    geometryError createOptixBoundingBox( RTcontext &context, RTgeometry &geometry );
	
    float boundingBox_max[3];
    float boundingBox_min[3];
	char path_to_ptx_boundingBox[512];
	char path_to_ptx_intersect[512];
	char comment[GEOM_CMT_LENGTH];
	simMode mode;

	Geometry_Params *paramsPtr;
	Geometry_ReducedParams *reducedParamsPtr;
	// optix variables associated with geometry
	RTvariable		params;
    RTgeometryinstance instance;
	RTgeometry		geometry;
	RTvariable		geometryID;
	RTvariable		l_mode;
	RTvariable		l_materialListLength;

	RTprogram		geometry_intersection_program;
	RTprogram		geometry_boundingBox_program;

	bool update; // if true, the variables of the OptiX-instance have to be update before next call to rtTrace

	virtual geometryError reduceParams(void);

  private:
	

  public:
	geometry_type type;

    /* standard constructor */
    Geometry()
	{
	  materialList=NULL;
	  materialListLength=0;
	}
    /* Constructor in case, list length is known */
    Geometry(int length)
	{
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i=0;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	}
	/* Destruktor */
	virtual ~Geometry()
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
	}

	void setPathToPtxIntersect(char* path);
	const char* getPathToPtxIntersect(void);
	void setPathToPtxBoundingBox(char* path);
	char* getPathToPtxBoundingBox(void);
	void setType(geometry_type type);
	geometry_type getType(void);

//	int getID(void);
//	void setID(int ID);
	void setBoundingBox_min(float *box_min);
	void setBoundingBox_max(float *box_max);
	Material* getMaterial(int index);
	float* getBoundingBox_min(void);
	float* getBoundingBox_max(void);
	geometryError setMaterial(Material *oMaterialPtr, int index);
	geometryError setMaterialListLength(int length);
	geometryError createMaterial(int index);
	geometryError setComment(char *ptrIn);
	char *getComment(void);
	
	virtual geometryError createOptixInstance(RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda);
	virtual geometryError updateOptixInstance(RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda);
    //virtual geometryError  trace(rayStruct &ray);
	virtual double intersect(rayStruct *ray);
	virtual double intersect(diffRayStruct *ray);
	virtual gaussBeam_t intersect(gaussBeamRayStruct *ray);
	virtual geometryError hit(rayStruct &ray, double t);
	virtual geometryError hit(diffRayStruct &ray, double t);
	virtual geometryError hit(gaussBeamRayStruct &ray, gaussBeam_t t);
	virtual Geometry_Params* getParamsPtr(void);
	virtual geometryError setParams(Geometry_Params *params);
	virtual geometryError createCPUSimInstance(double lambda, simMode mode);
	virtual geometryError updateCPUSimInstance(double lambda, simMode mode);
	virtual geometryError processParseResults(GeometryParseParamStruct &parseResults_Geom, int geomID);
	virtual geometryError parseXml(pugi::xml_node &geometry, simMode l_mode, vector<Geometry*> &geomVec);
};


#endif

