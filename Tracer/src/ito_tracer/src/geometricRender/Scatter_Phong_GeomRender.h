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

/**\file Scatter_Phong_GeomRender.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCATTER_PHONG_GEOMRENDER_H
#define SCATTER_PHONG_GEOMRENDER_H

#include "../Scatter_Phong.h"
#include <sampleConfig.h>
#include <stdio.h>
//#include "Group.h"
#include "Scatter_Phong_GeomRender_hit.h"

#define PATH_TO_HIT_SCATTER_PHONG_GEOMRENDER "_Scatter_Phong_GeomRender"

///* declare class */
///**
//  *\class   ScatLambert2D_scatParams
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
//class ScatLambert2D_GeomRender_scatParams: public ScatLambert2D_scatParams
//{
//public:
////	double TIR; // Total Integrated Scatter of surface
//};

/* declare class */
/**
  *\class   Scatter_Phong_GeomRender
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
class Scatter_Phong_GeomRender: public Scatter_Phong
{
	protected:
//		ScatLambert2D_GeomRender_params reducedParams; // reduced parameter set for the ray trace ( on GPU )
//		ScatLambert2D_scatParams *fullParamsPtr; // complete parameter set

  public:
    /* standard constructor */
    Scatter_Phong_GeomRender()
	{
        reducedParams.coefLambertian=0;
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s", PATH_TO_HIT_SCATTER_PHONG_GEOMRENDER );
		this->setPathToPtx(path_to_ptx);
	}

	//ScatterError setFullParams(ScatLambert2D_scatParams* ptrIn);
	//ScatLambert2D_scatParams* getFullParams(void);
	//ScatterError setReducedParams(ScatLambert2D_params* paramsIn);
	//ScatLambert2D_params* getReducedParams(void);
	//void set_nRefr2(double nIn);
	//double get_nRefr2(void);
    ScatterError createOptiXInstance(double lambda, char** path_to_ptx_in);
	//ScatterError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	ScatterError createCPUSimInstance(double lambda);
	//void setPathToPtx(char* path);
	//char* getPathToPtx(void);
	void hit(rayStruct &ray, Mat_hitParams hitParams);
};

#endif


