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

#ifndef MATERIALVOLUMESCATTERBOX_HIT_H
#define MATERIALVOLUMESCATTERBOX_HIT_H

#include "rayTracingMath.h"
#include "randomGenerator.h"
#include "VolumeScattererBox_intersect.h"
//#include "differentialRayTracing/Material_DiffRays_hit.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

/* declare class */
/**
  *\class   MatVolumeScatterBox_params 
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
class MatVolumeScatterBox_params
{
public:
	double n1; // refractive index1
	double n2; // refractive index2
	double meanFreePath;
	double g; // anisotropy factor
	double absorptionCoeff; // absorption coefficient
	int maxNrBounces;
	double2 aprtRadius;
	double thickness;
	double3 root;
	double3 tilt;
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
inline RT_HOSTDEVICE bool hitVolumeScatterBox(rayStruct &ray, Mat_hitParams hitParams, MatVolumeScatterBox_params params, double t_hit, int geomID, bool coat_reflected)
{
	Mat_hitParams debugHitParams=hitParams;
	MatVolumeScatterBox_params debugParams=params;

	ray.position=ray.position+t_hit*ray.direction;
	ray.opl=ray.opl+t_hit*ray.nImmersed;

	if (coat_reflected)
	{
		ray.direction=reflect(ray.direction,hitParams.normal);
		return 1;
	}

	VolumeScattererBox_ReducedParams l_boxParams;
	l_boxParams.root=params.root;
	l_boxParams.tilt=params.tilt;
	double3 l_normal=make_double3(0,0,1);
	rotateRay(&l_normal,params.tilt);
	l_boxParams.normal=l_normal;
	l_boxParams.thickness=params.thickness;
	l_boxParams.apertureRadius=params.aprtRadius;
	double t_surf;

	// calc snells law upon entry in material
	Mat_hitParams l_hitParams=calcHitParamsVolumeScattererBox(ray.position, l_boxParams);
	// calc Snells Law
	if (ray.nImmersed==params.n1)
	{
		if (calcSnellsLaw(&(ray.direction),l_hitParams.normal, ray.nImmersed, params.n2))
		{
			ray.nImmersed=params.n2;
		}
		else
		{
			return 1; // if we had TIR we do not need to calc volume scatter
		}
	}
	else if (ray.nImmersed==params.n2)
	{
		if (calcSnellsLaw(&(ray.direction), l_hitParams.normal, ray.nImmersed, params.n1))
		{
			ray.nImmersed=params.n1;
		}
		else
		{
			return 1; // if we had TIR we do not need to calc volume scatter
		}
	}
	else
	{
		// some error mechanism
		return 0;
	}

	// if the ray was inside 
	double l_t=0;
	uint32_t x1[5];
	RandomInit(ray.currentSeed, x1); // seed random generator
	for (unsigned int iBounce=0; iBounce<params.maxNrBounces; iBounce++)
	{
		for (unsigned int i=0; i<50; ++i) // in principle, this should be an endless loop, we inserted a maximum number of iterations for performance reasons
		{
			// randomly generate free path until next scatter
			// the idea is to generate a random distance betwen zero and 10 times the mean free path
			// then, the probability of this distance is compared to a random number between 0 and 1
			l_t=Random(x1)*params.meanFreePath*20;
			double l_prob=exp(-l_t/params.meanFreePath);
			double l_threshold=Random(x1);
			if (l_prob>l_threshold)
					break;
		}

		t_surf=intersectRayVolumeScattererBox(ray.position, ray.direction, l_boxParams);
		// if we have a surface hit
		if ( (l_t>t_surf) && (t_surf>0) )
		{
			l_hitParams=calcHitParamsVolumeScattererBox(ray.position, l_boxParams);
			// update position to surface
			ray.position=ray.position+t_surf*ray.direction;
			ray.opl=ray.opl+t_surf*ray.nImmersed;
			ray.flux=ray.flux*exp(-params.absorptionCoeff*t_surf);
			// calc Snells Law
			if (ray.nImmersed==params.n1)
			{
				if (calcSnellsLaw(&(ray.direction),l_hitParams.normal, ray.nImmersed, params.n2))
				{
					ray.nImmersed=params.n2;
					ray.currentSeed=x1[0];
					return 1; // if we have left the volume we return
				}
			}
			else if (ray.nImmersed==params.n2)
			{
				if (calcSnellsLaw(&(ray.direction), l_hitParams.normal, ray.nImmersed, params.n1))
				{
					ray.nImmersed=params.n1;
					ray.currentSeed=x1[0];
					return 1; // if we have left the volume we return
				}
			}
			else
			{
				// some error mechanism
				ray.currentSeed=x1[0];
				return 0;
			}
			
		}

		// update ray
		ray.position=ray.position+l_t*ray.direction;
		ray.opl=ray.opl+ray.nImmersed*t_hit;
		ray.flux=ray.flux*exp(-params.absorptionCoeff*l_t);
		// change ray direction
		double3 l_tilt;
		l_tilt.x=(Random(x1)-0.5)*2*params.g;
		l_tilt.y=(Random(x1)-0.5)*2*params.g;
		l_tilt.z=(Random(x1)-0.5)*2*params.g;
		// backscatter
		if (params.g<0)
			ray.direction=-ray.direction;
		rotateRay(&ray.direction, l_tilt);
	}

	t_surf=intersectRayVolumeScattererBox(ray.position, ray.direction, l_boxParams);
	ray.position=ray.position+t_surf*ray.direction;
	ray.opl=ray.opl+t_surf*ray.nImmersed;
	ray.flux=ray.flux*exp(-params.absorptionCoeff*t_surf);

	l_hitParams=calcHitParamsVolumeScattererBox(ray.position, l_boxParams);
	// calc Snells Law upon exit from volume
	if (ray.nImmersed==params.n1)
	{
		if (calcSnellsLaw(&(ray.direction),l_hitParams.normal, ray.nImmersed, params.n2))
			ray.nImmersed=params.n2;
	}
	else if (ray.nImmersed==params.n2)
	{
		if (calcSnellsLaw(&(ray.direction), l_hitParams.normal, ray.nImmersed, params.n1))
			ray.nImmersed=params.n1;
	}
	else
	{
		// some error mechanism
		ray.currentSeed=x1[0];
		return 0;
	}
	// in principle we need to take care of the case that we get TIR upon exit from the volume...
	ray.currentSeed=x1[0];
	return 1;
}

#endif


