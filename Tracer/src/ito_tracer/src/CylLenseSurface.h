/*
Copyright (C) 2012 ITO university stuttgart

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; If not, see <http://www.gnu.org/licenses/>.

*/

/**\file CylLenseSurface.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef CYLLENSE_H
  #define CYLLENSE_H
  
/* include header of basis class */
#include "Geometry.h"
#include "CylLenseSurface_intersect.h"

/* declare class */
/**
  *\class   CylLenseSurface_Params
  *\ingroup Geometry
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
class CylLenseSurface_Params : public Geometry_Params
{
  public:
 	  double3 root;
	  double3 orientation;
	  double radius;
//	  double rotNormal; // rotation of geometry around its normal
//	  double thickness; 
	  double2 aptHalfWidth;
	  ApertureType aptType;
	  //int geometryID;
};

/* declare class */
/**
  *\class   CylLenseSurface
  *\ingroup Geometry
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
class CylLenseSurface : public Geometry
{
  protected:
	CylLenseSurface_Params *paramsPtr;
	CylLenseSurface_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

private:
	/* copy constructor */
	CylLenseSurface(const CylLenseSurface &obj)
	{
	}
	/* declare copy operator */
	CylLenseSurface& operator=(const CylLenseSurface& op)
	{
	}

  public:
    /* standard constructor */
    CylLenseSurface()
	{
	  paramsPtr=new CylLenseSurface_Params();
	  reducedParamsPtr=new CylLenseSurface_ReducedParams();
	  materialList=NULL;
	  materialListLength=0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_CylPipe.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_CYLLENSESURF;
	}
    /* Constructor in case length of list is already known */
    CylLenseSurface(int length)
	{
	  paramsPtr=new CylLenseSurface_Params();
	  reducedParamsPtr=new CylLenseSurface_ReducedParams();
	  materialList=new Material*[length];
	  int i;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_CylLenseSurface.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_CYLLENSE;
	}
	/* Destruktor */
	~CylLenseSurface()
	{
	  if (materialList!=NULL)
	  {
		  // delete all the materials attached to this geometry
		  int i;
		  for (i=0;i<materialListLength;i++)
		  {
			  if (materialList[i]!=NULL)
			  {
				  delete materialList[i];
			  }
		  }
		  delete materialList;
		  materialList = NULL;
	  }
	  if (paramsPtr != NULL)
	  {
		delete paramsPtr;
		paramsPtr=NULL;
	  }
	  if ( reducedParamsPtr != NULL)
	  {
	    delete reducedParamsPtr;
		reducedParamsPtr = NULL;
	  }
	}
	
    Geometry_Params* getParamsPtr(void);
	geometryError setParams(Geometry_Params *params);
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	double intersect(rayStruct *ray);
//	gaussBeam_t intersect(gaussBeamRayStruct *ray);
	geometryError hit(rayStruct &ray, double t);
//	geometryError hit(gaussBeamRayStruct &ray, gaussBeam_t t);
	geometryError processParseResults(GeometryParseParamStruct &parseResults_Geom, int geomID);

};

#endif
