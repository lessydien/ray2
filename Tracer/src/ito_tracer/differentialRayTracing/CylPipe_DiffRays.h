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

/**\file CylPipe_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef CYLPIPE_DIFFRAYS_H
  #define CYLPIPE_DIFFRAYS_H
  
/* include header of basis class */
#include "../CylPipe.h"
#include "CylPipe_DiffRays_intersect.h"

///* declare class */
///**
//  *\class   CylPipe_DiffRays_Params 
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
//class CylPipe_DiffRays_Params : public CylPipe_Params
//{
//  public:
// 	  double3 root;
//	  double3 orientation;
//	  double2 radius;
//	  double rotNormal; // rotation of geometry around its normal
//	  double thickness;   
//	  //int geometryID;
//};

/* declare class */
/**
  *\class   CylPipe_DiffRays 
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
class CylPipe_DiffRays : public CylPipe
{
  protected:
//	CylPipe_DiffRays_Params *paramsPtr;
	CylPipe_DiffRays_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

private:
	/* copy constructor */
	CylPipe_DiffRays(const CylPipe_DiffRays &obj)
	{
	}
	/* declare copy operator */
	CylPipe_DiffRays& operator=(const CylPipe_DiffRays& op)
	{
	}

  public:
    /* standard constructor */
    CylPipe_DiffRays()
	{
//	  paramsPtr=new CylPipe_DiffRays_Params();
	  reducedParamsPtr=new CylPipe_DiffRays_ReducedParams();
	  materialList=NULL;
	  materialListLength=0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_CylPipe_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_CYLPIPE;
	}
    /* Constructor in case length of list is already known */
    CylPipe_DiffRays(int length)
	{
//	  paramsPtr=new CylPipe_DiffRays_Params();
	  reducedParamsPtr=new CylPipe_DiffRays_ReducedParams();
	  materialList=new Material*[length];
	  int i;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_CylPipe_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_CYLPIPE;
	}
	/* Destruktor */
	~CylPipe_DiffRays()
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
	  if (paramsPtr != NULL)
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
