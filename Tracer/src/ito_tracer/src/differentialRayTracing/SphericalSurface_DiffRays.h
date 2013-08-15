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

/**\file SphericalSurface_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SPHERICALSURFACE_DIFFRAYS_H
  #define SPHERICALSURFACE_DIFFRAYS_H
  
/* include header of basis class */
#include "../SphericalSurface.h"
#include "SphericalSurface_DiffRays_intersect.h"
#include <optix.h>

///* declare class */
///**
//  *\class   SphericalSurface_DiffRays_params
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
//class SphericalSurface_DiffRays_Params : public SphericalSurface_Params
//{
//	public:
// 	  double3 centre;
//	  double3 orientation;
//	  double2 curvatureRadius;
//	  double2 apertureRadius;
//	  double rotNormal; // rotation of geometry around its normal
//	  ApertureType apertureType;
//	  //int geometryID;
//};

/* declare class */
/**
  *\class   SphericalSurface_DiffRays
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
class SphericalSurface_DiffRays : public SphericalSurface
{
  protected:
//	SphericalSurface_DiffRays_Params	*paramsPtr;
	SphericalSurface_DiffRays_ReducedParams	*reducedParamsPtr;

	geometryError reduceParams(void);

  private:
  	/* declare copy operator */
	SphericalSurface_DiffRays& operator=(const SphericalSurface_DiffRays& op)
	{
	}
	/* copy constructor */
	SphericalSurface_DiffRays(const SphericalSurface_DiffRays &obj)
	{
	}

  public:
    /* standard constructor */
    SphericalSurface_DiffRays()
	{
//	  paramsPtr=new SphericalSurface_DiffRays_Params();
	  reducedParamsPtr=new SphericalSurface_DiffRays_ReducedParams();
	  materialList = NULL;
	  materialListLength = 0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_SphericalSurface_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_SPHERICALSURF;
	}
    /* Constructor with known list length */
    SphericalSurface_DiffRays(int length)
	{
//	  paramsPtr=new SphericalSurface_DiffRays_Params();
	  reducedParamsPtr=new SphericalSurface_DiffRays_ReducedParams();
	  materialList=new Material*[length];
	  int i;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_SphericalSurface_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_SPHERICALSURF;
	}
	/* Destruktor */
	~SphericalSurface_DiffRays()
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
				  materialList[i]=NULL;
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
	  if (reducedParamsPtr != NULL)
	  {
	    delete reducedParamsPtr;
		reducedParamsPtr = NULL;
	  }
	}

	double intersect(diffRayStruct *ray);
	geometryError hit(diffRayStruct &ray, double t);
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );

};

#endif
