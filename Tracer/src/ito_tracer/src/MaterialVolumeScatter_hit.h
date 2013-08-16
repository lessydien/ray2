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

/**\file MaterialVolumeScatter_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MATERIALVOLUMESCATTER_HIT_H
#define MATERIALVOLUMESCATTER_HIT_H

#include "rayTracingMath.h"
#include "randomGenerator.h"
//#include "differentialRayTracing/Material_DiffRays_hit.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

/* declare class */
/**
  *\class   MatVolumeScatter_params 
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
class MatVolumeScatter_params
{
public:
	double n1; // refractive index1
	double n2; // refractive index2
	double meanFreePath;
	double g; // anisotropy factor
	double absorptionCoeff; // absorption coefficient
	int maxNrBounces;
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
inline RT_HOSTDEVICE bool hitVolumeScatter(rayStruct &ray, Mat_hitParams hitParams, MatVolumeScatter_params params, double t_hit, int geomID, bool coat_reflected)
{
	Mat_hitParams debugHitParams=hitParams;
	MatVolumeScatter_params debugParams=params;
	// if the ray was inside 
	double l_t=0;
	bool l_surfaceHit=true;
	// if the ray already was inside the volumscatterer
	if (ray.currentGeometryID == VOLSCATTER_REFRIDX)
	{
		uint32_t x1[5];
		RandomInit(ray.currentSeed, x1); // seed random generator
		for (unsigned int i=0; i<50; ++i) // in principle, this should be an endless loop, we inserted a maximum number of iterations for performance reasons
		{
			// randomly generate free path until next scatter
			// the idea is to generate a random distance betwen zero and 10 times the mean free path
			// then, the probability of this distance is compared to a random number between 0 and 1
			l_t=Random(x1)*params.meanFreePath*20;
			double l_prob=exp(-l_t/params.meanFreePath);
			double l_threshold=Random(x1);
			if (l_prob>l_threshold)
			{
				ray.currentSeed=x1[4];
				break;
			}
		}
		// if we have a volume hit
		if (l_t<t_hit)
		{
			// update ray
			ray.position=ray.position+l_t*ray.direction;
			ray.opl=ray.opl+ray.nImmersed*t_hit;
			// change ray direction
			double3 l_tilt;
			l_tilt.x=(Random(x1)-0.5)*2*params.g;
			l_tilt.y=(Random(x1)-0.5)*2*params.g;
			l_tilt.z=(Random(x1)-0.5)*2*params.g;
			rotateRay(&ray.direction, l_tilt);

			l_surfaceHit=false;
		}
		ray.currentSeed=x1[4]; // save seed for next round
	}
	if (l_surfaceHit) // if we have a surface hit
	{
		// update ray position 
		ray.position=ray.position+t_hit*ray.direction;
		ray.currentGeometryID=geomID;
		ray.opl=ray.opl+ray.nImmersed*t_hit;
		// if the coating wants us to have reflection, we do reflection here instead of refraction
		if (coat_reflected)
			ray.direction=reflect(ray.direction,hitParams.normal);
		else
		{
			if (ray.nImmersed==params.n1)
			{
				if (calcSnellsLaw(&(ray.direction),hitParams.normal, ray.nImmersed, params.n2))
					ray.nImmersed=params.n2;
			}
			else if (ray.nImmersed==params.n2)
			{
				if (calcSnellsLaw(&(ray.direction), hitParams.normal, ray.nImmersed, params.n1))
					ray.nImmersed=params.n1;
			}
			else
			{
				// some error mechanism
				return 0;
			}
			if ( ray.currentGeometryID==VOLSCATTER_REFRIDX)
				ray.currentGeometryID=geomID; // signal that the ray is on the surface of geometry
			else
				ray.currentGeometryID=VOLSCATTER_REFRIDX; // signal that the ray is inside the volum scatterer
		}
	}
	return 1;
}

#endif


