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

/**\file PlaneSurface.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef PLANESURFACE_H
  #define PLANESURFACE_H
  
/* include header of basis class */
#include "Geometry.h"
#include "PlaneSurface_intersect.h"

/* declare class */
/**
  *\class   PlaneSurface_Params
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
class PlaneSurface_Params : public Geometry_Params
{
  public:
    double3 root;
    double3 normal;
   //int geometryID;
};

/* declare class */
/**
  *\class   PlaneSurface
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
class PlaneSurface : public Geometry
{
  protected:
	PlaneSurface_Params *paramsPtr;
	PlaneSurface_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

private:
	/* copy constructor */
	PlaneSurface(const PlaneSurface &obj)
	{
	}
	/* declare copy operator */
	PlaneSurface& operator=(const PlaneSurface& op)
	{
	}

  public:
    /* standard constructor */
    PlaneSurface()
	{
	  paramsPtr=new PlaneSurface_Params();
	  reducedParamsPtr=new PlaneSurface_ReducedParams();
	  materialList=NULL;
	  materialListLength=0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_PlaneSurface.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_PLANESURF;
	}
    /* Constructor in case length of list is already known */
    PlaneSurface(int length)
	{
	  paramsPtr=new PlaneSurface_Params();
	  reducedParamsPtr=new PlaneSurface_ReducedParams();
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i=0;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_PlaneSurface.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_PLANESURF;
	}
	/* Destruktor */
	~PlaneSurface()
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
	
    Geometry_Params* getParamsPtr(void);
	geometryError setParams(Geometry_Params *paramsIn);//PlaneSurface_Params *params);
	double intersect(rayStruct *ray);
//	double intersect(diffRayStruct *ray);
	gaussBeam_t intersect(gaussBeamRayStruct *ray);
	geometryError hit(rayStruct &ray, double t);
//	geometryError hit(diffRayStruct &ray, double t);
	geometryError hit(gaussBeamRayStruct &ray, gaussBeam_t t);
	geometryError processParseResults(GeometryParseParamStruct &parseResults_Geom, int geomID);
	geometryError parseXml(pugi::xml_node &geometry, simMode l_mode, vector<Geometry*> &geomVec);
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );
};

#endif
