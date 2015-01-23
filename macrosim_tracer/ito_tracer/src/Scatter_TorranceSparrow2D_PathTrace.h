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

/**\file Scatter_TorranceSparrow2D_PathTrace.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCATTER_TORRANCESPARROW2D_PATHTRACE_H
#define SCATTER_TORRANCESPARROW2D_PATHTRACE_H

#include "Scatter_TorranceSparrow2D.h"
#include <sampleConfig.h>
#include <stdio.h>
//#include "Group.h"
#include "Scatter_TorranceSparrow2D_PathTrace_hit.h"

#define PATH_TO_HIT_SCATTER_TORRANCESPARROW2D_PATHTRACE "_Scatter_TorranceSparrow2D_PathTrace"

/* declare class */
/**
  *\class   ScatTorranceSparrow2D_PathTrace_scatParams
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
class ScatTorranceSparrow2D_PathTrace_scatParams: public ScatTorranceSparrow2D_scatParams
{
public:
	ScatTorranceSparrow2D_PathTrace_scatParams()
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
		srcAreaHalfWidth=make_double2(0,0);
		srcAreaRoot=make_double3(0,0,0);
		srcAreaTilt=make_double3(0,0,0);
		srcAreaType=AT_UNKNOWNATYPE;
	}
	~ScatTorranceSparrow2D_PathTrace_scatParams()
	{
	}
	double2 srcAreaHalfWidth;
	double3 srcAreaRoot;
	double3 srcAreaTilt;
	ApertureType srcAreaType;
};

/* declare class */
/**
  *\class   Scatter_TorranceSparrow2D_PathTrace
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
class Scatter_TorranceSparrow2D_PathTrace: public Scatter_TorranceSparrow2D
{
	protected:
		ScatTorranceSparrow2D_PathTrace_params reducedParams; // reduced parameter set for the ray trace ( on GPU )
		ScatTorranceSparrow2D_PathTrace_scatParams *fullParamsPtr; // complete parameter set

  public:
    /* standard constructor */
    Scatter_TorranceSparrow2D_PathTrace()
	{
		reducedParams.Kdl=0;
		reducedParams.Ksl=0;
		reducedParams.Ksp=0;
		reducedParams.sigmaXsl=0;
		reducedParams.sigmaXsp=0;
		reducedParams.scatAxis=make_double3(0,0,0);
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s", PATH_TO_HIT_SCATTER_TORRANCESPARROW2D_PATHTRACE );
		this->setPathToPtx(path_to_ptx);
	}

	ScatterError setFullParams(ScatTorranceSparrow2D_PathTrace_scatParams* ptrIn);
	ScatTorranceSparrow2D_PathTrace_scatParams* getFullParams(void);
	ScatterError setReducedParams(ScatTorranceSparrow2D_PathTrace_params* paramsIn);
	ScatTorranceSparrow2D_PathTrace_params* getReducedParams(void);
	ScatterError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	ScatterError createCPUSimInstance(double lambda);
	void hit(rayStruct &ray, Mat_hitParams hitParams);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	ScatterError processParseResults(MaterialParseParamStruct &parseResults_Mat);
};

#endif


