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

/**\file Scatter.h
* \brief 
* 
*           
* \author Mauch
*/

/**
 *\defgroup Scatter
 */

#ifndef SCATTER_H
  #define SCATTER_H

#include <optix.h>
#include <optix_math.h>
#include "rayData.h"
#include "GlobalConstants.h"
#include "Scatter_hit.h"
#include "stdlib.h"
#include "MaterialParams.h"
#include "pugixml.hpp"
#include <stdio.h>

typedef enum 
{
  SCAT_ERROR,
  SCAT_NO_ERROR
} ScatterError;

/* declare class */
/**
  *\class   Scatter_params
  *\ingroup Scatter
  *\brief   base class of the full params of the scatters defined in this application
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
class Scatter_Params
{
public:
	Scatter_Params()
	{
		impAreaHalfWidth=make_double2(0,0);
		impAreaRoot=make_double3(0,0,0);
		impAreaTilt=make_double3(0,0,0);
		impAreaType=AT_INFTY;
		type=ST_NOSCATTER;
	}
	~Scatter_Params()
	{
	}

	double2 impAreaHalfWidth;
	double3 impAreaRoot;
	double3 impAreaTilt;
	ApertureType impAreaType;
	ScatterType type;
};

/* declare class */
/**
  *\class   Scatter
  *\brief   base class of all scatters defined in this application
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
class Scatter
{
  protected:
    char* path_to_ptx;
//	Scatter_Params *fullParamsPtr;
//	Scatter_ReducedParams reducedParams;
		
  public:
    bool update;
    
	/* standard constructor */
    Scatter()
	{
//		this->fullParamsPtr=new Scatter_Params();
		path_to_ptx=(char*)malloc(512*sizeof(char));
		sprintf( this->path_to_ptx, "" );
	}
	/* destructor */
	virtual ~Scatter()
	{
		delete path_to_ptx;
		//if (this->fullParamsPtr != NULL)
		//{
		//	delete (this->fullParamsPtr);
		//	this->fullParamsPtr=NULL;
		//}
	}
	virtual void setPathToPtx(char* path);
	virtual char* getPathToPtx(void);
//    virtual ScatterError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	virtual ScatterError createCPUSimInstance(double lambda);
	virtual ScatterError createOptiXInstance(double lambda, char** path_to_ptx_in);
	virtual ScatterError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	virtual void hit(rayStruct &ray, Mat_hitParams hitParams);
	virtual void hit(diffRayStruct &ray, Mat_hitParams hitParams);
	virtual void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	virtual ScatterError setFullParams(Scatter_Params* ptrIn);
	virtual Scatter_Params* getFullParams(void);
	virtual ScatterError setReducedParams(Scatter_ReducedParams* ptrIn);
	virtual Scatter_ReducedParams* getReducedParams(void);
	virtual ScatterError processParseResults(MaterialParseParamStruct &parseResults_Mat);
	virtual ScatterError parseXml(pugi::xml_node &node, SimParams simParams);
	virtual bool checkParserError(char *msg);
};

#endif

