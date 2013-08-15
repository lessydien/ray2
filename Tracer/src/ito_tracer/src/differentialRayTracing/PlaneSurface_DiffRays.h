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

/**\file PlaneSurface_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef PLANESURFACE_DIFFRAYS_H
  #define PLANESURFACE_DIFFRAYS_H
  
/* include header of basis class */
#include "../PlaneSurface.h"
#include "PlaneSurface_DiffRays_intersect.h"
#include "Material_DiffRays_hit.h"

///* declare class */
///**
//  *\class   PlaneSurface_DiffRays_Params 
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
//class PlaneSurface_DiffRays_Params : public PlaneSurface_Params
//{
//  public:
//	  // is exactly the same as PlaneSurface_Params
//};

/* declare class */
/**
  *\class   PlaneSurface_DiffRays
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
class PlaneSurface_DiffRays : public PlaneSurface
{
  protected:
//	PlaneSurface_DiffRays_Params *paramsPtr;
	PlaneSurface_DiffRays_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

private:
	/* copy constructor */
	PlaneSurface_DiffRays(const PlaneSurface_DiffRays &obj)
	{
	}
	/* declare copy operator */
	PlaneSurface_DiffRays& operator=(const PlaneSurface_DiffRays& op)
	{
	}

  public:
    /* standard constructor */
    PlaneSurface_DiffRays()
	{
//	  paramsPtr=new PlaneSurface_DiffRays_Params();
	  reducedParamsPtr=new PlaneSurface_DiffRays_ReducedParams();
	  materialList=NULL;
	  materialListLength=0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_PlaneSurface_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_PLANESURF;
	}
    /* Constructor in case length of list is already known */
    PlaneSurface_DiffRays(int length)
	{
//	  paramsPtr=new PlaneSurface_DiffRays_Params();
	  reducedParamsPtr=new PlaneSurface_DiffRays_ReducedParams();
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i=0;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_PlaneSurface_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_PLANESURF;
	}
	/* Destruktor */
	~PlaneSurface_DiffRays()
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
	  if ( paramsPtr != NULL)
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
	
//    Geometry_Params* getParamsPtr(void);
//	geometryError setParams(Geometry_Params *paramsIn);//PlaneSurface_DiffRays_Params *params);
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );
	double intersect(diffRayStruct *ray);
	geometryError hit(diffRayStruct &ray, double t);
};

#endif
