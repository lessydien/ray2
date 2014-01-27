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

/**\file CadObject.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef CADOBJECT_H
  #define CADOBJECT_H
  
/* include header of basis class */
#include <nvModel.h>
#include "Geometry.h"
#include "cadObject_intersect.h"

/* declare class */
/**
  *\class   CadObject_Params
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
class CadObject_Params : public Geometry_Params
{
  public:
    double3 root;
    double3 normal;
   //int geometryID;
};

/* declare class */
/**
  *\class   CadObject
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
class CadObject : public Geometry
{
  protected:
	CadObject_Params *paramsPtr;
	CadObject_ReducedParams *reducedParamsPtr;
	nv::Model* model;
	RTbuffer vertex_buffer_obj; //!> this is where the buffers for the vertices of OptiX simulation will be stored
	RTbuffer index_buffer_obj; //!> this is where the buffers for the indices OptiX simulation will be stored

	geometryError reduceParams(void);

private:
	/* copy constructor */
	CadObject(const CadObject &obj)
	{
	}
	/* declare copy operator */
	CadObject& operator=(const CadObject& op)
	{
	}

  public:
    /* standard constructor */
    CadObject()
	{
	  paramsPtr=new CadObject_Params();
	  reducedParamsPtr=new CadObject_ReducedParams();
	  materialList=NULL;
	  materialListLength=0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_CadObject.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_CadObject.cu.ptx" );
	  type=GEOM_PLANESURF;
	}
    /* Constructor in case length of list is already known */
    CadObject(int length)
	{
	  paramsPtr=new CadObject_Params();
	  reducedParamsPtr=new CadObject_ReducedParams();
	  materialList=new Material*[length];
	  // init the pointers to zero
	  int i=0;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_CadObject.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, "ITO-MacroSim_generated_CadObject.cu.ptx" );
	  type=GEOM_PLANESURF;
	}
	/* Destruktor */
	~CadObject()
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
	geometryError setParams(Geometry_Params *paramsIn);//CadObject_Params *params);
	double intersect(rayStruct *ray);
//	double intersect(diffRayStruct *ray);
	geometryError hit(rayStruct &ray, double t);
//	geometryError hit(diffRayStruct &ray, double t);
	geometryError processParseResults(GeometryParseParamStruct &parseResults_Geom, int geomID);
	geometryError parseXml(pugi::xml_node &geometry, SimParams simParams, vector<Geometry*> &geomVec);
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, SimParams simParams, double lambda );
};

#endif
