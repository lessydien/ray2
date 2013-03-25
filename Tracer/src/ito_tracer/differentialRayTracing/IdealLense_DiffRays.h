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

/**\file IdealLense_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef IDEALLENSE_DIFFRAYS_H
  #define IDEALLENSE_DIFFRAYS_H
  
/* include header of basis class */
#include "../IdealLense.h"
#include "IdealLense_DiffRays_intersect.h"

///* declare class */
///**
//  *\class   IdealLense_DiffRays_Params 
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
//class IdealLense_DiffRays_Params : public IdealLense_Params
//{
//  public:
//   double3 root;
//   double3 normal;
//   double2 apertureRadius;
//   double rotNormal; // rotation of geometry around its normal
//   ApertureType apertureType;
//   //int geometryID;
//};

/* declare class */
/**
  *\class   IdealLense_DiffRays
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
/* declare class */
class IdealLense_DiffRays : public IdealLense
{
  protected:
//	IdealLense_DiffRays_Params *paramsPtr;
	IdealLense_DiffRays_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

private:
	/* copy constructor */
	IdealLense_DiffRays(const IdealLense_DiffRays &obj)
	{
	}
	/* declare copy operator */
	IdealLense_DiffRays& operator=(const IdealLense_DiffRays& op)
	{
	}

  public:
    /* standard constructor */
    IdealLense_DiffRays()
	{
	  paramsPtr=new IdealLense_Params();
	  reducedParamsPtr=new IdealLense_DiffRays_ReducedParams();
	  materialList=NULL;
	  materialListLength=0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_IdealLense_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_IDEALLENSE;
	}
    /* Constructor in case length of list is already known */
    IdealLense_DiffRays(int length)
	{
	  paramsPtr=new IdealLense_Params();
	  reducedParamsPtr=new IdealLense_DiffRays_ReducedParams();
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i=0;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_IdealLense_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_PLANESURF;
	}
	/* Destruktor */
	~IdealLense_DiffRays()
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
	
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );
	double intersect(diffRayStruct *ray);
	geometryError hit(diffRayStruct &ray, double t);

};

#endif
