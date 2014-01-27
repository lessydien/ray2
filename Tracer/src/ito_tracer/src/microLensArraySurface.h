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

/**\file MicroLensArraySurface.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MICROLENSARRAYSURFACE_H
  #define MICROLENSARRAYSURFACE_H
  
/* include header of basis class */
#include "Geometry.h"
#include "MicroLensArraySurface_intersect.h"
#include <optix.h>

/* declare class */
/**
  *\class   MicroLensArraySurface_params
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
class MicroLensArraySurface_Params : public virtual Geometry_Params
{
	public:
 	  double3 root;
	  double3 normal;
	  double microLensRadius;
	  double microLensPitch;
	  double microLensAptRad;
	  MicroLensAptType microLensAptType;
};

/* declare class */
/**
  *\class   MicroLensArraySurface
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
class MicroLensArraySurface : public virtual Geometry
{
  protected:
	MicroLensArraySurface_Params	*paramsPtr;
	MicroLensArraySurface_ReducedParams	*reducedParamsPtr;

	geometryError reduceParams(void);

  private:


  public:
    /* standard constructor */
    MicroLensArraySurface()
	{
	  paramsPtr=new MicroLensArraySurface_Params();
	  reducedParamsPtr=new MicroLensArraySurface_ReducedParams();
	  materialList = NULL;
	  materialListLength = 0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_MicroLensArraySurface.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_MicroLensArraySurface.cu.ptx" );
	  type=GEOM_SPHERICALSURF;
	}
    /* Constructor with known list length */
    MicroLensArraySurface(int length)
	{
	  paramsPtr=new MicroLensArraySurface_Params();
	  reducedParamsPtr=new MicroLensArraySurface_ReducedParams();
	  materialList=new Material*[length];
	  int i;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_MicroLensArraySurface.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_MicroLensArraySurface.cu.ptx" );
	  type=GEOM_SPHERICALSURF;
	}
	/* Destruktor */
	~MicroLensArraySurface()
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
