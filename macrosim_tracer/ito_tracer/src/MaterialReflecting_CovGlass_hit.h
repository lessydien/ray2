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

/**\file MaterialReflecting.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MATERIALREFLECTING_COVGLASS_HIT_H
#define MATERIALREFLECTING_COVGLASS_HIT_H

#include "rayTracingMath.h"
#include "Material_hit.h"

/* declare class */
/**
  *\class   MatReflecting_CovGlass_params 
  *\ingroup Material
  *\brief   reduced set of params that is calculated before the actual tracing from the full set of params. This parameter set will be loaded onto the GPU if the tracing is done there
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
class MatReflecting_CovGlass_params
{
public:
	double r; // amplitude reflection coefficient
	double t; // amplitude transmission coefficient
	int geomID; // only rays that come from the geometry with this ID will be reflected
};

/**
 * \detail hitReflecting_CovGlass 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] rayStruct &ray, Mat_hitParams hitParams, MatReflecting_CovGlass_params params, double t_hit, int geometryID
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitReflecting_CovGlass(rayStruct &ray, Mat_hitParams hitParams, MatReflecting_CovGlass_params params, double t_hit, int geometryID)
{
    ray.position = ray.position + t_hit * ray.direction;
	ray.opl=ray.opl+ray.nImmersed*t_hit;
	if (params.geomID == ray.currentGeometryID)
	{
		ray.direction=reflect(ray.direction,hitParams.normal);
		ray.flux=ray.flux*params.r;
		ray.depth++;
	}
	else
	{
		ray.flux=ray.flux*params.t;
	}
	ray.currentGeometryID=geometryID; // in any case the ray is sitting on the cover glass no
	return true;
}

#endif


