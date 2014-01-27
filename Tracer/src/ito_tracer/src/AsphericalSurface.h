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

/**\file AsphericalSurface.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef ASPHERICALSURFACE_H
  #define ASPHERICALSURFACE_H
  
/* include header of basis class */
#include "Geometry.h"
#include "AsphericalSurface_intersect.h"

/* declare class */
/**
  *\class   AsphericalSurface_Params 
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
class AsphericalSurface_Params : public Geometry_Params
{
  public:
  double3 vertex;
  double3 orientation;
  double k; // conic constant
  double c; // 1/(radius of curvature)
  double c2;
  double c4;
  double c6;
  double c8;
  double c10;
  double c12;
  double c14;
  double c16;
//  double rotNormal; // rotation of geometry around its normal
  //int    geometryID;
};

/* declare class */
/**
  *\class            AsphericalSurface
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
class AsphericalSurface : public Geometry
{
  protected:
    AsphericalSurface_Params *paramsPtr;
	AsphericalSurface_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

  private:
	/* declare copy operator */
	AsphericalSurface& operator=(const AsphericalSurface& op)
	{
	}
	/* declare copy constructor */
	AsphericalSurface(const AsphericalSurface &obj)
	{
	}

  public:
    /* standard constructor */
    AsphericalSurface()
	{
	  paramsPtr=new AsphericalSurface_Params();
	  reducedParamsPtr=new AsphericalSurface_ReducedParams();
	  materialList = NULL;
	  materialListLength = 0;
	  type=GEOM_ASPHERICALSURF;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_AsphericalSurface.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_AsphericalSurface.cu.ptx" );
	}
    /* Constructor with known list length */
    AsphericalSurface(int length)
	{
	  paramsPtr=new AsphericalSurface_Params();
	  reducedParamsPtr=new AsphericalSurface_ReducedParams();
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  type=GEOM_ASPHERICALSURF;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_AsphericalSurface.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_AsphericalSurface.cu.ptx" );
	}
	/* Destruktor */
	~AsphericalSurface()
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

	Geometry_Params* getParamsPtr(void);
	geometryError setParams(Geometry_Params *params);
	double intersect(rayStruct *ray);
//	gaussBeam_t intersect(gaussBeamRayStruct *ray);
	geometryError hit(rayStruct &ray, double t);
//	geometryError hit(gaussBeamRayStruct &ray, gaussBeam_t t);
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	geometryError processParseResults(GeometryParseParamStruct &parseResults_Geom, int geomID);
	geometryError parseXml(pugi::xml_node &geometry, SimParams simParams, vector<Geometry*> &geomVec);

};

#endif
