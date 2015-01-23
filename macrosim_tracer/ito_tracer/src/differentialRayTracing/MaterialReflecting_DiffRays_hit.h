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

#ifndef MATERIALREFLECTING_DIFFRAYS_HIT_H
#define MATERIALREFLECTING_DIFFRAYS_HIT_H

#include "../rayTracingMath.h"
#include "Material_DiffRays_hit.h"
#include "../MaterialReflecting_hit.h"

/* declare class */
/**
  *\class   MatReflecting_DiffRays_params
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
class MatReflecting_DiffRays_params : public MatReflecting_params
{
public:
//	double r; // amplitude reflection coefficient
};

/**
 * \detail hitReflecting_DiffRays 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitReflecting_DiffRays(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geomID, bool coat_reflected)
{
	Mat_DiffRays_hitParams hittest=hitParams;
	// update ray position
	ray.position=ray.position+t_hit*ray.direction;
	ray.currentGeometryID=geomID;
	ray.opl=ray.opl+ray.nImmersed*t_hit;
	double3 oldRayDirection=ray.direction; // save ray direction for use in refractDifferentialData()
	// calc differential ray stuff
	// calc new wavefront radii
	double2 newWavefrontRad;
	newWavefrontRad.x=(ray.wavefrontRad.x-t_hit*ray.nImmersed);
	newWavefrontRad.y=(ray.wavefrontRad.y-t_hit*ray.nImmersed);
	// calc local flux from wavefront radii
	ray.flux=ray.flux*abs(ray.wavefrontRad.x/newWavefrontRad.x)*abs(ray.wavefrontRad.y/newWavefrontRad.y);
	// save new wavefront radii
	ray.wavefrontRad=newWavefrontRad;

	// see Diss. O. Stolz: "Differentielles Ray Tracing für spezielle Beleuchtungssysteme", Uni Erlangen 2010
	// transform tha data of the incoming ray
	double3 P_r, T_r;
	double2 radius_r;
	double torsion_r;
	double testPhi=acos(dot(ray.direction,hitParams.normal));
	transformDifferentialData(ray.direction, ray.mainDirY, ray.wavefrontRad, hitParams.normal, P_r, T_r, radius_r, torsion_r );

	// transform the data of the surface
	double3 PBar_r, TBar_r;
	double2 radiusBar_r;
	double torsionBar_r;
	transformDifferentialData(ray.direction, hitParams.mainDirY, hitParams.mainRad, hitParams.normal, PBar_r, TBar_r, radiusBar_r, torsionBar_r );

    // if the coating wants us to have reflection, we do reflection here instead of refraction
	double mu;
	double3 PPrime_r, TPrime_r;
	double2 radiusPrime_r;
	double torsionPrime_r;
	// do the reflection
	ray.direction=reflect(ray.direction,hitParams.normal);
	mu=1;
	// calc the data of the reflected ray
	reflectDifferentialData(oldRayDirection, hitParams.normal, ray.direction,  P_r, radius_r, radiusBar_r, torsion_r, torsionBar_r, mu, PPrime_r, TPrime_r, radiusPrime_r, torsionPrime_r, ray.flux);

	// transform the data of the refracted ray into its local system
	invTransformDifferentialData(PPrime_r, TPrime_r, radiusPrime_r, torsionPrime_r, ray.mainDirX, ray.mainDirY, ray.wavefrontRad);

    // adjust flux due to projections on interface
	ray.flux=ray.flux*abs(dot(oldRayDirection,hitParams.normal))/abs(dot(ray.direction,hitParams.normal));

	return 1;
}

#endif


