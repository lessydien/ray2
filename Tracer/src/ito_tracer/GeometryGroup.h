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

/**\file GeometryGroup.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef GEOMETRYGROUP_H
  #define GEOMETRYGROUP_H

#include <optix.h>
#include "Geometry.h"
#include "rayData.h"

typedef enum 
{
  GEOMGROUP_NO_ERR,
  GEOMGROUP_INDEX_OUT_OF_RANGE,
  GEOMGROUP_LISTCREATION_ERR,
  GEOMGROUP_NOGEOM_ERR,
  GEOMGROUP_ADDGEOM_UNKNOWNGEOM_ERR,
  GEOMGROUP_ERR

} geometryGroupError;

/* declare class */
/**
  *\class   GeometryGroup
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
class GeometryGroup
{
  protected:
    Geometry** geometryList;
	unsigned int geometryListLength;
	simMode mode;
	geometryGroupError findClosestGeometry();

	/* OptiX variables */
	RTgeometrygroup OptiX_geometrygroup;
	RTacceleration		acceleration;
    RTtransform			transforms;


  private:
    /* copy constructor */
    GeometryGroup(const GeometryGroup& obj)
	{
		// defined private to prevent the user from using this constructor
	}
	/* declare copy operator */
	GeometryGroup& operator=(const GeometryGroup& op)
	{
		// defined private to prevent the user from using ths operator
	}

  public:
    /* standard constructor */
    GeometryGroup()
	{
	  geometryList=NULL;
	  geometryListLength=0;
	}
    /* Constructor in case list length is known */
    GeometryGroup(unsigned int length)
	{
	  geometryList=new Geometry*[length];
	  // init the pointers to zero
	  unsigned int i=0;
	  for (i=0;i<length;i++)
	  {
		  geometryList[i]=NULL;
	  }
	  geometryListLength = length;
	}
	/* Destruktor */
	~GeometryGroup()
	{
	  /* destroy all geometries attached to this group */
	  unsigned int i;
	  for (i=0;i<geometryListLength;i++)
	  {
		  if (geometryList[i] != NULL)
		  {
			delete geometryList[i];
			geometryList[i]=NULL;
		  }
	  }
	  delete geometryList;
	  geometryList = NULL;
	}

    geometryGroupError  trace();
	int getGeometryListLength( void );
	geometryGroupError setGeometry(Geometry* oGeometry, unsigned int index);
	Geometry* getGeometry(unsigned int index);
    geometryGroupError createOptixInstance(RTcontext &context, RTgroup &group, unsigned int  index, simMode mode, double lambda);
	geometryGroupError createOptixInstance(RTcontext &context, RTselector &selector, unsigned int index, simMode mode, double lambda);
    geometryGroupError updateOptixInstance(RTcontext &context, RTgroup &group, unsigned int  index, simMode mode, double lambda);
	geometryGroupError updateOptixInstance(RTcontext &context, RTselector &selector, unsigned int index, simMode mode, double lambda);
	geometryGroupError trace(rayStruct &ray);
	geometryGroupError trace(diffRayStruct &ray);
	geometryGroupError trace(gaussBeamRayStruct &ray);
	geometryGroupError setGeometryListLength(unsigned int length);
	geometryGroupError createGeometry(unsigned int index);
	geometryGroupError createCPUSimInstance(double lambda, simMode mode );
	geometryGroupError updateCPUSimInstance(double lambda, simMode mode );
};

#endif

