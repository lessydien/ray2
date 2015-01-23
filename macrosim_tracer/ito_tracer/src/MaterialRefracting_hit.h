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

/**\file MaterialRefracting_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MATERIALREFRACTING_HIT_H
#define MATERIALREFRACTING_HIT_H

#include "rayTracingMath.h"
//#include "differentialRayTracing/Material_DiffRays_hit.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

/* declare class */
/**
  *\class   MatRefracting_params 
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
class MatRefracting_params
{
public:
	double n1; // refractive index1
	double n2; // refractive index2
	//double t; // amplitude transmission coefficient
	//double r; // amplitude reflection coefficient
};

/**
 * \detail hitRefracting 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] rayStruct &ray, Mat_hitParams hitParams, MatRefracting_params params, double t_hit, int geomID, bool coat_reflected
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitRefracting(rayStruct &ray, Mat_hitParams hitParams, MatRefracting_params params, double t_hit, int geomID, bool coat_reflected)
{
	Mat_hitParams debugHitParams=hitParams;
	MatRefracting_params debugParams=params;
	// update ray
	ray.position=ray.position+t_hit*ray.direction;
	ray.opl=ray.opl+ray.nImmersed*t_hit;
	ray.currentGeometryID=geomID;

	// if the coating wants us to have reflection, we do reflection here instead of refraction
	if (coat_reflected)
		ray.direction=reflect(ray.direction,hitParams.normal);
	else
	{
		if (ray.nImmersed==params.n1)
		{
			// if we had refraction, we need to change the immersion of the ray
			if (calcSnellsLaw(&(ray.direction),hitParams.normal, ray.nImmersed, params.n2))
				ray.nImmersed=params.n2;
		}
		else if (ray.nImmersed==params.n2)
		{
			// if we had refraction, we need to change the immersion of the ray
			if (calcSnellsLaw(&(ray.direction),hitParams.normal, ray.nImmersed, params.n1))
				ray.nImmersed=params.n1;
		}
		else
		{
			// some error mechanism
			ray.running=false; // stop ray
			return 0;
		}
	}

	return 1;
}

#endif


