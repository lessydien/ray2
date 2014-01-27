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

/**\file StopArraySurface.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef STOPARRAYSURFACE_H
  #define STOPARRAYSURFACE_H
  
/* include header of basis class */
#include "Geometry.h"
#include "stopArraySurface_intersect.h"
#include <optix.h>

/* declare class */
/**
  *\class   StopArraySurface_params
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
class StopArraySurface_params : public virtual Geometry_Params
{
	public:
 	  double3 root;
	  double3 normal;
	  double2 microStopPitch;
	  double2 microStopRad;
	  ApertureType microStopType;
};

/* declare class */
/**
  *\class   StopArraySurface
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
class StopArraySurface : public virtual Geometry
{
  protected:
	StopArraySurface_params	*paramsPtr;
	StopArraySurface_ReducedParams	*reducedParamsPtr;

	geometryError reduceParams(void);

  private:


  public:
    /* standard constructor */
    StopArraySurface()
	{
	  paramsPtr=new StopArraySurface_params();
	  reducedParamsPtr=new StopArraySurface_ReducedParams();
	  materialList = NULL;
	  materialListLength = 0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_StopArraySurface.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_StopArraySurface.cu.ptx" );
	  type=GEOM_SPHERICALSURF;
	}
    /* Constructor with known list length */
    StopArraySurface(int length)
	{
	  paramsPtr=new StopArraySurface_params();
	  reducedParamsPtr=new StopArraySurface_ReducedParams();
	  materialList=new Material*[length];
	  int i;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_StopArraySurface.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_StopArraySurface.cu.ptx" );
	  type=GEOM_SPHERICALSURF;
	}
	/* Destruktor */
	~StopArraySurface()
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

    Geometry_Params* getParamsPtr(void);
	geometryError setParams(Geometry_Params *params);
	double intersect(rayStruct *ray);
//	gaussBeam_t intersect(gaussBeamRayStruct *ray);
	geometryError hit(rayStruct &ray, double t);
//	geometryError hit(gaussBeamRayStruct &ray, gaussBeam_t t);
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	geometryError parseXml(pugi::xml_node &geometry, SimParams simParams, vector<Geometry*> &geomVec);
};

#endif
