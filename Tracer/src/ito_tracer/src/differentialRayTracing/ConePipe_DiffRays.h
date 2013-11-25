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

/**\file ConePipe_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef CONEPIPE_DIFFRAYS_H
  #define CONEPIPE_DIFFRAYS_H
  
/* include header of basis class */
#include "../ConePipe.h"
#include "ConePipe_DiffRays_intersect.h"

///* declare class */
///**
//  *\class   ConePipe_DiffRays_Params 
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
//class ConePipe_DiffRays_Params : public ConePipe_Params
//{
//  public:
// 	  double3 root; // starting point of the cone segment
//	  double3 coneEnd; // end point of the cone. where the sidewalls would meet in one point
//	  double3 orientation; // orientation of the symmetrie axis of the cone
//	  double2 cosTheta; // half opening angle of the cone in x and y
//	  double thickness; // length of the cone segment
//	  double rotNormal; // rotation of geometry around its normal
//	  //int geometryID;
//};

/* declare class */
/**
  *\class   ConePipe_DiffRays 
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
class ConePipe_DiffRays : public ConePipe
{
  protected:
//	ConePipe_DiffRays_Params *paramsPtr;
	ConePipe_DiffRays_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

private:
	/* copy constructor */
	ConePipe_DiffRays(const ConePipe_DiffRays &obj)
	{
	}
	/* declare copy operator */
	ConePipe_DiffRays& operator=(const ConePipe_DiffRays& op)
	{
	}

  public:
    /* standard constructor */
    ConePipe_DiffRays()
	{
//	  paramsPtr=new ConePipe_DiffRays_Params();
	  reducedParamsPtr=new ConePipe_DiffRays_ReducedParams();
	  materialList=NULL;
	  materialListLength=0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_ConePipe_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_CONEPIPE;
	}
    /* Constructor in case length of list is already known */
    ConePipe_DiffRays(int length)
	{
//	  paramsPtr=new ConePipe_DiffRays_Params();
	  reducedParamsPtr=new ConePipe_DiffRays_ReducedParams();
	  materialList=new Material*[length];
	  int i;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_ConePipe_DiffRays.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_CONEPIPE;
	}
	/* Destruktor */
	~ConePipe_DiffRays()
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
	  if (reducedParamsPtr != NULL)
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
