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

/**\file Group.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef GROUP_H
  #define GROUP_H

#include <optix.h>
#include "GeometryGroup.h"
#include "rayData.h"
#include "GlobalConstants.h"
#include <stdio.h>

typedef enum 
{
  GROUP_ERR,
  GROUP_NO_ERR,
  GROUP_LISTCREATION_ERR,
  GROUP_NOGEOMGROUP_ERR
} groupError;

/* declare class */
/**
  *\class   Group
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
class Group
{
  protected:
    char path_to_ptx_visit[512];
	int geometryGroupListLength;
	GeometryGroup** geometryGroupList;
	SimParams mode;
	groupError findClosestGeometryGroup();

	/* OptiX variables */
	RTgroup OptiX_group; // OptiX OptiX_group. This is where all the geometryInstances are put into in nonsequential mode
	RTselector selector; // OptiX selector. This is where all the geometryInstances are put into in nonsequential mode
	RTacceleration		top_level_acceleration;
	RTvariable			top_object;
	RTprogram			l_visit_program; // visit program to select the geometries to intersect sequentially in sequential mode

  public:
	/* standard constructor */
    Group()
	{
	  geometryGroupList=NULL;
	  geometryGroupListLength=0;
	  sprintf( this->path_to_ptx_visit, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_Selector_Visit_Program.cu.ptx" );
	}
    /* Constructor in case length of list is already known*/
    Group(int length)
	{
	  geometryGroupList=new GeometryGroup*[length];
	  for (int i=0;i<length;i++)
		  geometryGroupList[i]=NULL;
	  geometryGroupListLength = length;
	  sprintf( this->path_to_ptx_visit, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_Selector_Visit_Program.cu.ptx" );
	}
	/* Destruktor */
	~Group()
	{
	  /* delete scene */
	  int i;
	  for (i=0; i<geometryGroupListLength; i++)
	  {
	    delete geometryGroupList[i];
	  }
	  delete geometryGroupList;
	  geometryGroupList = 0;
	}

	groupError setGeometryGroupListLength(int length);
    groupError trace(rayStruct &ray);
	groupError trace(diffRayStruct &ray);
	groupError trace(gaussBeamRayStruct &ray);
	int getGeometryGroupListLength( void );
	groupError setGeometryGroup(GeometryGroup* oGeometryGroup, int index);
	GeometryGroup* getGeometryGroup(int index);
	groupError createOptixInstance(RTcontext &context, SimParams simParams, double lambda);
	groupError updateOptixInstance(RTcontext &context, SimParams simParams, double lambda);
	groupError createGeometryGroup(int index);
	groupError createCPUSimInstance(double lambda, SimParams simParams );
	groupError updateCPUSimInstance(double lambda, SimParams simParams );
};

#endif

