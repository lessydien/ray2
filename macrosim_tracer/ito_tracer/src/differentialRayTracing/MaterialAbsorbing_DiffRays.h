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

/**\file MaterialAbsorbing_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALABSORBING_DIFFRAYS_H
#define MATERIALABSORBING_DIFFRAYS_H

#include "Material_DiffRays.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "../MaterialAbsorbing.h"
#include "MaterialAbsorbing_DiffRays_hit.h"

#define PATH_TO_HIT_ABSORBING_DIFFRAYS "macrosim_tracer_generated_hitFunctionAbsorbing"

/* declare class */
/**
  *\class   MaterialAbsorbing_DiffRays
  *\ingroup Material
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
class MaterialAbsorbing_DiffRays: public MaterialAbsorbing
{
	protected:

  public:
    /* standard constructor */
    MaterialAbsorbing_DiffRays()
	{
		/* set ptx path for OptiX calculations */
		//path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_ABSORBING_DIFFRAYS );
		//this->setPathToPtx(path_to_ptx);
	}
    ~MaterialAbsorbing_DiffRays()
	{
		//delete path_to_ptx;
	}

	void hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID);
};
#endif