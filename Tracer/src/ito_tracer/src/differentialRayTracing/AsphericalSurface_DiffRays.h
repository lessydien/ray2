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

/**\file AsphericalSurface_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef ASPHERICALSURFACE_DIFFRAYS_H
  #define ASPHERICALSURFACE_DIFFRAYS_H
  
/* include header of basis class */
#include "../Geometry.h"
#include "../AsphericalSurface.h"
#include "AsphericalSurface_DiffRays_intersect.h"

///* declare class */
///**
//  *\class   AsphericalSurface_DiffRays_Params 
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
//class AsphericalSurface_DiffRays_Params : public Geometry_Params
//{
//  public:
//  double3 vertex;
//  double3 orientation;
//  double k;
//  double c;
//  double c2;
//  double c4;
//  double c6;
//  double c8;
//  double c10;
//  double rotNormal; // rotation of geometry around its normal
//  //int    geometryID;
//
//  ApertureType apertureType;
//  double2 apertureRadius;
//};

/* declare class */
/**
  *\class            AsphericalSurface_DiffRays 
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
class AsphericalSurface_DiffRays : public AsphericalSurface
{
  protected:
//    AsphericalSurface_DiffRays_Params *paramsPtr;
	AsphericalSurface_DiffRays_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

  private:
	/* declare copy operator */
	AsphericalSurface_DiffRays& operator=(const AsphericalSurface_DiffRays& op)
	{
	}
	/* declare copy constructor */
	AsphericalSurface_DiffRays(const AsphericalSurface_DiffRays &obj)
	{
	}

  public:
    /* standard constructor */
    AsphericalSurface_DiffRays()
	{
//	  paramsPtr=new AsphericalSurface_DiffRays_Params();
	  reducedParamsPtr=new AsphericalSurface_DiffRays_ReducedParams();
	  materialList = NULL;
	  materialListLength = 0;
	  type=GEOM_ASPHERICALSURF;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_AsphericalSurface_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	}
    /* Constructor with known list length */
    AsphericalSurface_DiffRays(int length)
	{
//	  paramsPtr=new AsphericalSurface_DiffRays_Params();
	  reducedParamsPtr=new AsphericalSurface_DiffRays_ReducedParams();
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  type=GEOM_ASPHERICALSURF;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_AsphericalSurface_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	}
	/* Destruktor */
	~AsphericalSurface_DiffRays()
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
		  materialList = 0;
	  }
	  if ( paramsPtr != NULL)
	  {
		delete paramsPtr;
		paramsPtr=NULL;
	  }
	  if ( reducedParamsPtr != NULL)
	  {
		  delete reducedParamsPtr;
		  reducedParamsPtr=NULL;
	  }
	}

	double intersect(diffRayStruct *ray);
	geometryError hit(diffRayStruct &ray, double t);
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );

};

#endif
