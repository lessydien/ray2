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

/**\file Scatter_TorranceSparrow1D_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCATTER_TORRANCESPARROW_DIFFRAYS_H
#define SCATTER_TORRANCESPARROW_DIFFRAYS_H

#include "../Scatter_TorranceSparrow1D.h"
#include <sampleConfig.h>
#include <stdio.h>
//#include "Group.h"
#include "Scatter_TorranceSparrow1D_DiffRays_hit.h"
#include "Material_DiffRays_hit.h"

#define PATH_TO_HIT_SCATTER_TORRANCESPARROW1D_DIFFRAYS "_Scatter_TorranceSparrow1D_DiffRays"

///* declare class */
///**
//  *\class   ScatTorranceSparrow1D_scatParams
//  *\brief   full set of params
//  *
//  *         
//  *
//  *         \todo
//  *         \remarks           
//  *         \sa       NA
//  *         \date     04.01.2011
//  *         \author  Mauch
//  *
//  */
//class ScatTorranceSparrow1D_DiffRays_scatParams: public ScatTorranceSparrow1D_scatParams
//{
//public:
//	//double Kdl; // coefficient of diffuse lobe
//	//double Ksl; // coefficient of specular lobe
//	//double Ksp; // coefficient of specular peak
//	//double sigmaXsl; // width parameter of specular lobe
//	//double sigmaXsp; // width parameter of specular peak
//	//double3 scatAxis;
//};

/* declare class */
/**
  *\class   Scatter_TorranceSparrow1D_DiffRays
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
class Scatter_TorranceSparrow1D_DiffRays: public Scatter_TorranceSparrow1D
{
	protected:
//		ScatTorranceSparrow1D_DiffRays_params reducedParams; // reduced parameter set for the ray trace ( on GPU )
//		ScatTorranceSparrow1D_scatParams *fullParamsPtr; // complete parameter set

  public:
    /* standard constructor */
    Scatter_TorranceSparrow1D_DiffRays()
	{
		reducedParams.Kdl=0;
		reducedParams.Ksl=0;
		reducedParams.Ksp=0;
		reducedParams.sigmaXsl=0;
		reducedParams.sigmaXsp=0;
		reducedParams.scatAxis=make_double3(0,0,0);
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s", PATH_TO_HIT_SCATTER_TORRANCESPARROW1D_DIFFRAYS );
		this->setPathToPtx(path_to_ptx);
	}

	//ScatterError setFullParams(ScatTorranceSparrow1D_scatParams* ptrIn);
	//ScatTorranceSparrow1D_scatParams* getFullParams(void);
	//ScatterError setReducedParams(ScatTorranceSparrow1D_params* paramsIn);
	//ScatTorranceSparrow1D_params* getReducedParams(void);
	//void set_nRefr2(double nIn);
	//double get_nRefr2(void);
    ScatterError createOptiXInstance(double lambda, char** path_to_ptx_in);
	//ScatterError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	ScatterError createCPUSimInstance(double lambda);
	//void setPathToPtx(char* path);
	//char* getPathToPtx(void);
	void hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams);
	//void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
};

#endif


