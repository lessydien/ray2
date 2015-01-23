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

/**\file Scatter_TorranceSparrow2D_GeomRender.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCATTER_TORRANCESPARROW2D_GEOMRENDER_H
#define SCATTER_TORRANCESPARROW2D_GEOMRENDER_H

#include "../Scatter_TorranceSparrow2D.h"
#include <sampleConfig.h>
#include <stdio.h>
//#include "Group.h"
#include "Scatter_TorranceSparrow2D_GeomRender_hit.h"

#define PATH_TO_HIT_SCATTER_TORRANCESPARROW2D_GEOMRENDER "_Scatter_TorranceSparrow2D_GeomRender"


/* declare class */
/**
  *\class   Scatter_TorranceSparrow2D_GeomRender
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
class Scatter_TorranceSparrow2D_GeomRender: public Scatter_TorranceSparrow2D
{
	protected:
//		ScatTorranceSparrow2D_params reducedParams; // reduced parameter set for the ray trace ( on GPU )
//		ScatTorranceSparrow2D_scatParams *fullParamsPtr; // complete parameter set

  public:
    /* standard constructor */
    Scatter_TorranceSparrow2D_GeomRender()
	{
		reducedParams.Kdl=0;
		reducedParams.Ksl=0;
		reducedParams.Ksp=0;
		reducedParams.sigmaXsl=0;
		reducedParams.sigmaXsp=0;
		reducedParams.scatAxis=make_double3(0,0,0);
		/* set ptx path for OptiX calculations */
		path_to_ptx[512];
		sprintf( path_to_ptx, "%s", PATH_TO_HIT_SCATTER_TORRANCESPARROW2D_GEOMRENDER );
		this->setPathToPtx(path_to_ptx);
	}

    ScatterError createOptiXInstance(char** path_to_ptx_in);
	ScatterError createCPUSimInstance();
	void hit(rayStruct &ray, Mat_hitParams hitParams);
	ScatterError processParseResults(MaterialParseParamStruct &parseResults_Mat);
};

#endif


