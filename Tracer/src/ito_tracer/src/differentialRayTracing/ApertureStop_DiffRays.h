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

/**\file ApertureStop_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef APERTURESTOP_DIFFRAYS_H
  #define APERTURESTOP_DIFFRAYS_H
  
/* include header of basis class */
#include "../Geometry.h"
#include "../ApertureStop.h"
#include "ApertureStop_DiffRays_intersect.h"

///* declare class */
///**
//  *\class   ApertureStop_DiffRays_Params 
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
//class ApertureStop_DiffRays_Params : public ApertureStop_Params
//{
//  public:
//   double3 root;
//   double3 normal;
//   double2 apertureRadius;
//   double2 apertureStopRadius;
//   double rotNormal; // rotation of geometry around its normal
//   ApertureType apertureType;
//   //int geometryID;
//};

/* declare class */
/**
  *\class   ApertureStop_DiffRays
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
class ApertureStop_DiffRays : public ApertureStop
{
  protected:
//	ApertureStop_Params *paramsPtr;
	ApertureStop_DiffRays_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

private:
	/* copy constructor */
	ApertureStop_DiffRays(const ApertureStop_DiffRays &obj)
	{
	}
	/* declare copy operator */
	ApertureStop_DiffRays& operator=(const ApertureStop_DiffRays& op)
	{
	}

  public:
    /* standard constructor */
    ApertureStop_DiffRays()
	{
//	  paramsPtr=new ApertureStop_DiffRays_Params();
	  reducedParamsPtr=new ApertureStop_DiffRays_ReducedParams();
	  materialList=NULL;
	  materialListLength=0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_ApertureStop_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_ApertureStop_DiffRays.cu.ptx" );
	  type=GEOM_APERTURESTOP;
	}
    /* Constructor in case length of list is already known */
    ApertureStop_DiffRays(int length)
	{
//	  paramsPtr=new ApertureStop_DiffRays_Params();
	  reducedParamsPtr=new ApertureStop_DiffRays_ReducedParams();
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i=0;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_ApertureStop_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_ApertureStop_DiffRays.cu.ptx" );
	  type=GEOM_PLANESURF;
	}
	/* Destruktor */
	~ApertureStop_DiffRays()
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
	
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	double intersect(diffRayStruct *ray);
	geometryError hit(diffRayStruct &ray, double t);

};

#endif
