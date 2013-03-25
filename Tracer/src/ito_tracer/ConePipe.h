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

/**\file ConePipe.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef CONEPIPE_H
  #define CONEPIPE_H
  
/* include header of basis class */
#include "Geometry.h"
#include "ConePipe_intersect.h"

/* declare class */
/**
  *\class   ConePipe_Params 
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
class ConePipe_Params : public Geometry_Params
{
  public:
 	  double3 root; // starting point of the cone segment
	  double3 coneEnd; // end point of the cone. where the sidewalls would meet in one point
	  double3 orientation; // orientation of the symmetrie axis of the cone
	  double2 cosTheta; // half opening angle of the cone in x and y
	  double thickness; // length of the cone segment
//	  double rotNormal; // rotation of geometry around its normal
	  //int geometryID;
};

/* declare class */
/**
  *\class   ConePipe 
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
class ConePipe : public Geometry
{
  protected:
	ConePipe_Params *paramsPtr;
	ConePipe_ReducedParams *reducedParamsPtr;

	geometryError reduceParams(void);

private:
	/* copy constructor */
	ConePipe(const ConePipe &obj)
	{
	}
	/* declare copy operator */
	ConePipe& operator=(const ConePipe& op)
	{
	}

  public:
    /* standard constructor */
    ConePipe()
	{
	  paramsPtr=new ConePipe_Params();
	  reducedParamsPtr=new ConePipe_ReducedParams();
	  materialList=NULL;
	  materialListLength=0;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_ConePipe.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_CONEPIPE;
	}
    /* Constructor in case length of list is already known */
    ConePipe(int length)
	{
	  paramsPtr=new ConePipe_Params();
	  reducedParamsPtr=new ConePipe_ReducedParams();
	  materialList=new Material*[length];
	  int i;
	  for (i=0;i<length;i++)
	  {
		  materialList[i]=NULL;
	  }
	  materialListLength = length;
	  sprintf( this->path_to_ptx_intersect, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_ConePipe.cu.ptx" );
	  sprintf( this->path_to_ptx_boundingBox, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, "ITO-MacroSim_generated_boundingBox.cu.ptx" );
	  type=GEOM_CONEPIPE;
	}
	/* Destruktor */
	~ConePipe()
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
	
    Geometry_Params* getParamsPtr(void);
	geometryError setParams(Geometry_Params *params);
	geometryError createOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );
	geometryError updateOptixInstance( RTcontext &context, RTgeometrygroup &geometrygroup, int index, simMode mode, double lambda );
	double intersect(rayStruct *ray);
//	gaussBeam_t intersect(gaussBeamRayStruct *ray);
//	geometryError hit(gaussBeamRayStruct &ray, gaussBeam_t t);
	geometryError hit(rayStruct &ray, double t);
	geometryError processParseResults(GeometryParseParamStruct &parseResults_Geom, int geomID);
	geometryError parseXml(pugi::xml_node &geometry, simMode l_mode, vector<Geometry*> &geomVec);

};

#endif
