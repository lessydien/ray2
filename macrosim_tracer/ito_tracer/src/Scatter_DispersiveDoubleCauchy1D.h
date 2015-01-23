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

/**\file Scatter_DispersiveDoubleCauchy1D.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCATTER_DISPERSIVEDOUBLECAUCHY1D_H
#define SCATTER_DISPERSIVEDOUBLECAUCHY1D_H

#include "Scatter_DoubleCauchy1D.h"
#include <sampleConfig.h>
#include <stdio.h>
//#include "Group.h"
#include "Scatter_DispersiveDoubleCauchy1D_hit.h"

#define PATH_TO_HIT_SCATTER_DISPDOUBLECAUCHY1D "_Scatter_DispersiveDoubleCauchy1D"

/* declare class */
/**
  *\class   ScatDispersiveDoubleCauchy1D_scatParams
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
class ScatDispersiveDoubleCauchy1D_scatParams: public ScatDoubleCauchy1D_scatParams
{
public:
	ScatDispersiveDoubleCauchy1D_scatParams()
	{
		impAreaHalfWidth=make_double2(0,0);
		impAreaRoot=make_double3(0,0,0);
		impAreaTilt=make_double3(0,0,0);
		impAreaType=AT_UNKNOWNATYPE;
		type=ST_NOSCATTER;
		a_gamma_sl=0;
		a_gamma_sp=0;
		c_gamma_sl=0;
		c_gamma_sp=0;
		a_k_sl=0;
		a_k_sp=0;
		c_k_sl=0;
		c_k_sp=0;
	}
	~ScatDispersiveDoubleCauchy1D_scatParams()
	{
	}
	double a_gamma_sl;
	double a_gamma_sp;
	double c_gamma_sl;
	double c_gamma_sp;
	double a_k_sl;
	double a_k_sp;
	double c_k_sl;
	double c_k_sp;
};

/* declare class */
/**
  *\class   Scatter_DispersiveDoubleCauchy1D
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
class Scatter_DispersiveDoubleCauchy1D: public Scatter_DoubleCauchy1D
{
	protected:
		ScatDispersiveDoubleCauchy1D_params reducedParams; // reduced parameter set for the ray trace ( on GPU )
		ScatDispersiveDoubleCauchy1D_scatParams *fullParamsPtr; // complete parameter set

  public:
    /* standard constructor */
    Scatter_DispersiveDoubleCauchy1D()
	{
		reducedParams.Ksl=0;
		reducedParams.Ksp=0;
		reducedParams.gammaXsl=0;
		reducedParams.gammaXsp=0;
		reducedParams.scatAxis=make_double3(0,0,0);
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" , PATH_TO_HIT_SCATTER_DISPDOUBLECAUCHY1D );
		this->setPathToPtx(path_to_ptx);
	}

	ScatterError setFullParams(ScatDispersiveDoubleCauchy1D_scatParams* ptrIn);
	ScatDispersiveDoubleCauchy1D_scatParams* getFullParams(void);
	ScatterError setReducedParams(ScatDispersiveDoubleCauchy1D_params* paramsIn);
	ScatDispersiveDoubleCauchy1D_params* getReducedParams(void);
	ScatterError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	ScatterError createCPUSimInstance(double lambda);
	void hit(rayStruct &ray, Mat_hitParams hitParams);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	ScatterError processParseResults(MaterialParseParamStruct &parseResults_Mat);
};

#endif


