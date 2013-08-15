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

/**\file Scatter_DoubleCauchy1D.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCATTER_DOUBLECAUCHY1D_H
#define SCATTER_DOUBLECAUCHY1D_H

#include "Scatter.h"
#include <sampleConfig.h>
#include <stdio.h>
//#include "Group.h"
#include "Scatter_DoubleCauchy1D_hit.h"

#define PATH_TO_HIT_SCATTER_DOUBLECAUCHY2D "_Scatter_DoubleCauchy1D"

/* declare class */
/**
  *\class   ScatDoubleCauchy1D_scatParams
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
class ScatDoubleCauchy1D_scatParams: public Scatter_Params
{
public:
	ScatDoubleCauchy1D_scatParams()
	{
		impAreaHalfWidth=make_double2(0,0);
		impAreaRoot=make_double3(0,0,0);
		impAreaTilt=make_double3(0,0,0);
		impAreaType=AT_UNKNOWNATYPE;
		type=ST_NOSCATTER;
		Ksl=0;
		Ksp=0;
		gammaXsl=0;
		gammaXsp=0;
		scatAxis=make_double3(1,0,0);
	}
	~ScatDoubleCauchy1D_scatParams()
	{
	}
	double Ksl; // coefficient of specular lobe
	double Ksp; // coefficient of specular peak
	double gammaXsl; // width parameter of specular lobe
	double gammaXsp; // width parameter of specular peak
	double3 scatAxis;
};

/* declare class */
/**
  *\class   Scatter_DoubleCauchy1D
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
class Scatter_DoubleCauchy1D: public Scatter
{
	protected:
		ScatDoubleCauchy1D_params reducedParams; // reduced parameter set for the ray trace ( on GPU )
		ScatDoubleCauchy1D_scatParams *fullParamsPtr; // complete parameter set

  public:
    /* standard constructor */
    Scatter_DoubleCauchy1D()
	{
		reducedParams.Ksl=0;
		reducedParams.Ksp=0;
		reducedParams.gammaXsl=0;
		reducedParams.gammaXsp=0;
		reducedParams.scatAxis=make_double3(0,0,0);
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" , PATH_TO_HIT_SCATTER_DOUBLECAUCHY2D );
		this->setPathToPtx(path_to_ptx);
	}

	ScatterError setFullParams(ScatDoubleCauchy1D_scatParams* ptrIn);
	ScatDoubleCauchy1D_scatParams* getFullParams(void);
	ScatterError setReducedParams(ScatDoubleCauchy1D_params* paramsIn);
	ScatDoubleCauchy1D_params* getReducedParams(void);
	ScatterError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	ScatterError createCPUSimInstance(double lambda);
	void hit(rayStruct &ray, Mat_hitParams hitParams);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	ScatterError processParseResults(MaterialParseParamStruct &parseResults_Mat);
};

#endif


