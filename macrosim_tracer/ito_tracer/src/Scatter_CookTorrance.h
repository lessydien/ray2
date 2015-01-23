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

/**\file Scatter_Phong.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCATTER_COOKTORRANCE_H
#define SCATTER_COOKTORRANCE_H

#include "Scatter.h"
#include <sampleConfig.h>
#include <stdio.h>
//#include "Group.h"
#include "Scatter_CookTorrance_hit.h"

#define PATH_TO_HIT_SCATTER_COOKTORRANCE "_Scatter_CookTorrance"

/* declare class */
/**
  *\class   ScatLambert2D_scatParams
  *\ingroup Scatter
  *\brief   full set of params
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
class ScatCookTorrance_scatParams: public Scatter_Params
{
public:
	ScatCookTorrance_scatParams
		
		
		
		
		()
	{
		impAreaHalfWidth=make_double2(0,0);
		impAreaRoot=make_double3(0,0,0);
		impAreaTilt=make_double3(0,0,0);
		impAreaType=AT_INFTY;
		type=ST_LAMBERT2D;
		coefLambertian=0;
		fresnelParam=0.8;
		roughnessFactor=1.0;
	}
	~ScatCookTorrance_scatParams()
	{
	}
	double coefLambertian; // Total Integrated Scatter of surface
	double fresnelParam;
	double roughnessFactor;
};

/* declare class */
/**
  *\class   Scatter_Phong
  *\ingroup Scatter
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
class Scatter_CookTorrance: public Scatter
{
	protected:
		ScatCookTorrance_params reducedParams; // reduced parameter set for the ray trace ( on GPU )
		ScatCookTorrance_scatParams *fullParamsPtr; // complete parameter set

  public:
    /* standard constructor */
    Scatter_CookTorrance()
	{
		this->fullParamsPtr=new ScatCookTorrance_scatParams();
		reducedParams.coefLambertian=0;
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s", PATH_TO_HIT_SCATTER_COOKTORRANCE );
		this->setPathToPtx(path_to_ptx);
	}
    ~Scatter_CookTorrance()
    {
        if (this->fullParamsPtr != NULL)
        {
            delete this->fullParamsPtr;
            this->fullParamsPtr=NULL;
        }
    }

	ScatterError setFullParams(ScatCookTorrance_scatParams* ptrIn);
	ScatCookTorrance_scatParams* getFullParams(void);
	ScatterError setReducedParams(ScatCookTorrance_params* paramsIn);
	ScatCookTorrance_params* getReducedParams(void);
	ScatterError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	ScatterError createCPUSimInstance(double lambda);
	virtual void hit(rayStruct &ray, Mat_hitParams hitParams);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	ScatterError processParseResults(MaterialParseParamStruct &parseResults_Mat);
	ScatterError parseXml(pugi::xml_node &geometry, SimParams simParams);
};

#endif


