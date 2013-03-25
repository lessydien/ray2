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

/**\file Coating.h
* \brief 
* 
*           
* \author Mauch
*/


/**
 *\defgroup Coating
 */

#ifndef COATING_H
  #define COATING_H

#include <optix.h>
#include <optix_math.h>
#include <optix_host.h>
#include "rayData.h"
#include "GlobalConstants.h"
#include "Coating_hit.h"
#include "stdlib.h"
#include "MaterialParams.h"
#include "pugixml.hpp"
#include <stdio.h>
//#include "CoatingLib.h"

typedef enum 
{
  COAT_ERROR,
  COAT_NO_ERROR
} CoatingError;

/* declare class */
/**
  *\class   Coating _FullParams
  *\ingroup Coating
  *\brief   base class of full set of params of coatings
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
class Coating_FullParams
{
public:
	Coating_FullParams()
	{
		type=CT_NOCOATING;
	}
	~Coating_FullParams()
	{
	}

	CoatingType type;
	//double t; // amplitude transmission coefficient
	//double r; // amplitude reflection coefficient
};

/* declare class */
/**
  *\class   Coating 
  *\ingroup Coating
  *\brief   base class of coatings
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
class Coating
{
  protected:
    char* path_to_ptx;
	Coating_FullParams *fullParamsPtr;
	Coating_ReducedParams *reducedParamsPtr;
	CoatingError calcCoatingCoeffs(double lambda, double3 normal, double3 direction);
			
  public:
    bool update;

    Coating()
	{
		fullParamsPtr=new Coating_FullParams();
		path_to_ptx=(char*)malloc(512*sizeof(char));
		sprintf( this->path_to_ptx, "" );
	}
	virtual ~Coating()
	{
		if (fullParamsPtr!=NULL)
		{
			delete fullParamsPtr;
			fullParamsPtr=NULL;
		}
	}
	void setPathToPtx(char* path);
	char* getPathToPtx(void);
	virtual CoatingError setFullParams(Coating_FullParams* ptrIn);
	virtual Coating_FullParams* getFullParams(void);
	virtual Coating_ReducedParams* getReducedParams(void);
//    virtual CoatingError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	virtual CoatingError createCPUSimInstance(double lambda);
	virtual CoatingError createOptiXInstance(double lambda, char** path_to_ptx_in);
	virtual CoatingError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	virtual bool hit(rayStruct &ray, Mat_hitParams hitParams);
	virtual bool hit(diffRayStruct &ray, Mat_hitParams hitParams);
	virtual bool hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	virtual CoatingError processParseResults(MaterialParseParamStruct &parseResults_Mat);
	virtual CoatingError parseXml(pugi::xml_node &node);
};

#endif

