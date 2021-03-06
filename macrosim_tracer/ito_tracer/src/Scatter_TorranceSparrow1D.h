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

/**\file Scatter_TorranceSparrow1D.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCATTER_TORRANCESPARROW_H
#define SCATTER_TORRANCESPARROW_H

#include "Scatter.h"
#include <sampleConfig.h>
#include <stdio.h>
//#include "Group.h"
#include "Scatter_TorranceSparrow1D_hit.h"

#define PATH_TO_HIT_SCATTER_TORRANCESPARROW1D "_Scatter_TorranceSparrow1D"

/* declare class */
/**
  *\class   ScatTorranceSparrow1D_scatParams
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
class ScatTorranceSparrow1D_scatParams: public Scatter_Params
{
public:
	ScatTorranceSparrow1D_scatParams()
	{
		impAreaHalfWidth=make_double2(0,0);
		impAreaRoot=make_double3(0,0,0);
		impAreaTilt=make_double3(0,0,0);
		impAreaType=AT_UNKNOWNATYPE;
		type=ST_NOSCATTER;
		Kdl=0;
		Ksl=0;
		Ksp=0;
		sigmaXsl=0;
		sigmaXsp=0;
		scatAxis=make_double3(1,0,0);
	}
	~ScatTorranceSparrow1D_scatParams()
	{
	}
	double Kdl; // coefficient of diffuse lobe
	double Ksl; // coefficient of specular lobe
	double Ksp; // coefficient of specular peak
	double sigmaXsl; // width parameter of specular lobe
	double sigmaXsp; // width parameter of specular peak
	double3 scatAxis;
};

/* declare class */
/**
  *\class   Scatter_TorranceSparrow1D
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
class Scatter_TorranceSparrow1D: public Scatter
{
	protected:
		ScatTorranceSparrow1D_params reducedParams; // reduced parameter set for the ray trace ( on GPU )
		ScatTorranceSparrow1D_scatParams *fullParamsPtr; // complete parameter set

  public:
    /* standard constructor */
    Scatter_TorranceSparrow1D()
	{
		reducedParams.Kdl=0;
		reducedParams.Ksl=0;
		reducedParams.Ksp=0;
		reducedParams.sigmaXsl=0;
		reducedParams.sigmaXsp=0;
		reducedParams.scatAxis=make_double3(0,0,0);
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s", PATH_TO_HIT_SCATTER_TORRANCESPARROW1D );
		this->setPathToPtx(path_to_ptx);
	}

	ScatterError setFullParams(ScatTorranceSparrow1D_scatParams* ptrIn);
	ScatTorranceSparrow1D_scatParams* getFullParams(void);
	ScatterError setReducedParams(ScatTorranceSparrow1D_params* paramsIn);
	ScatTorranceSparrow1D_params* getReducedParams(void);
	ScatterError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	ScatterError createCPUSimInstance(double lambda);
	void hit(rayStruct &ray, Mat_hitParams hitParams);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	ScatterError processParseResults(MaterialParseParamStruct &parseResults_Mat);
};

#endif


