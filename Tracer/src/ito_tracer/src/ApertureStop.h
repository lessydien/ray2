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

/**\file ApertureStop.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef APERTURESTOP_H
  #define APERTURESTOP_H
  
/* include header of basis class */
#include "Geometry.h"
#include "ApertureStop_intersect.h"

/* declare class */
/**
  *\class   ApertureStop_Params 
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
class ApertureStop_Params : public Geometry_Params
{
  public:
   double3 root;
   double3 normal;
   double2 apertureStopRadius;
//   double rotNormal; // rotation of geometry around its normal
   //int geometryID;
};

/* declare class */
/**
  *\class   ApertureStop
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
class ApertureStop : public Geometry
{
  protected:
	ApertureStop_Params *paramsPtr;
	ApertureStop_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

private:
	/* copy constructor */
	ApertureStop(const ApertureStop &obj)
	{
	}
	/* declare copy operator */
	ApertureStop& operator=(const ApertureStop& op)
	{
	}

  public:
    /* standard constructor */
    ApertureStop()
	{
	  paramsPtr=new ApertureStop_Params();
	  reducedParamsPtr=new ApertureStop_ReducedParams();
	  materialList=NULL;
	  materialListLength=0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_ApertureStop.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_APERTURESTOP;
	}
    /* Constructor in case length of list is already known */
    ApertureStop(int length)
	{
	  paramsPtr=new ApertureStop_Params();
	  reducedParamsPtr=new ApertureStop_ReducedParams();
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i=0;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_ApertureStop.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_PLANESURF;
	}
	/* Destruktor */
	~ApertureStop()
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
	geometryError setParams(Geometry_Params *paramsIn);//ApertureStop_Params *params);
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	double intersect(rayStruct *ray);
	gaussBeam_t intersect(gaussBeamRayStruct *ray);
	geometryError hit(rayStruct &ray, double t);
	geometryError hit(gaussBeamRayStruct &ray, gaussBeam_t t);
	geometryError processParseResults(GeometryParseParamStruct &parseResults_Geom, int geomID);
	geometryError parseXml(pugi::xml_node &geometry, TraceMode l_mode, vector<Geometry*> &geomVec);
};

#endif
