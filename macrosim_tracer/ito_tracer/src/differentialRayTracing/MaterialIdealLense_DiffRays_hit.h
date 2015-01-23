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

/**\file MaterialIdealLense_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MATERIALIDEALLENSE_DIFFRAYS_HIT_H
#define MATERIALIDEALLENSE_DIFFRAYS_HIT_H

#include "../rayTracingMath.h"
#include "Material_DiffRays_hit.h"
#include "../MaterialIdealLense_hit.h"

/* declare class */
/**
  *\class   MatIdealLense_DiffRays_params 
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
class MatIdealLense_DiffRays_params : public MatIdealLense_params
{
public:
	//double f; //!> focal length of ideal lense
	//double3 root; //!> root of the ideal lense
	//double3 orientation; //!> orientation of the ideal lense
	//double thickness; //!> thickness of ideal lense. This comes from the fact that the ray through the centre of the ideal lense hast to have a phase shift relative to the ray through the outermost corner of the aperture such that a perfect spherical wavefront appears behind the ideal lense
	double3 tilt;
};

/**
 * \detail hitIdealLense_DiffRays 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, MatIdealLense_params params, double t_hit, int geomID, bool coat_reflected
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitIdealLense_DiffRays(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, MatIdealLense_params params, double t_hit, int geomID, bool coat_reflected)
{
	MatIdealLense_params test=params;
	ray.position=ray.position+t_hit*ray.direction;
	ray.currentGeometryID=geomID;
	// apply OPL up to intersection with ideal lense
	ray.opl=ray.opl+ray.nImmersed*t_hit;

	// calc differential ray stuff on the incoming side of the hit point
	// calc new wavefront radii
	double2 newWavefrontRad;
	newWavefrontRad.x=(ray.wavefrontRad.x-t_hit*ray.nImmersed);
	newWavefrontRad.y=(ray.wavefrontRad.y-t_hit*ray.nImmersed);

	if ( (newWavefrontRad.x==0)||(newWavefrontRad.y==0) )
	{
		ray.flux=DOUBLE_MAX;
		newWavefrontRad.x=ray.lambda;
		newWavefrontRad.y=ray.lambda;
	}
	else
	{
		// calc local flux from wavefront radii
		ray.flux=ray.flux*abs(ray.wavefrontRad.x/newWavefrontRad.x)*abs(ray.wavefrontRad.y/newWavefrontRad.y);
	}
	// save new wavefront radii
	ray.wavefrontRad=newWavefrontRad;

	// calc differential ray stuff on outgoing side of hit point

	// here we use the same algorithm that is described in Diss. O. Stolz: "Differentielles Ray Tracing für spezielle Beleuchtungssysteme", Uni Erlangen 2010
	// for refraction of differential rays. The only difference is that we don't use snells law here but deflect the ray towards an ideal focal point...
	// transform tha data of the incoming ray
	double3 P_r, T_r;
	double2 radius_r;
	double torsion_r;
	transformDifferentialData(ray.direction, ray.mainDirY, ray.wavefrontRad, hitParams.normal, P_r, T_r, radius_r, torsion_r );

	// transform the data of the surface
	double3 PBar_r, TBar_r;
	double2 radiusBar_r;
	double torsionBar_r;
	transformDifferentialData(ray.direction, hitParams.mainDirY, hitParams.mainRad, hitParams.normal, PBar_r, TBar_r, radiusBar_r, torsionBar_r );

	// do the "refraction"
	// overwrite wavefront radii
	// wavefront is such that it becomes zero at the ideal image of the point where the incoming ray originated from
	// in principle it is the lense formula 1/b+1/g=1/f that lies the basis to this calculation
	// g has to be calculated from the wavefront at the incoming side and the direction of the incoming ray ( as
	// the ideal lense shows now image plane curvature, we need to calculate the distance of the origin of the 
	// incoming ray projected onto the normal of the ideal lense
	// As we might have astigmatic wavefronts, we need to do this for x and y respectively
	double g_x = -abs(dot(ray.direction,hitParams.normal))*ray.wavefrontRad.x;
	double g_y = -abs(dot(ray.direction,hitParams.normal))*ray.wavefrontRad.y;
	// now calculate ideal image point belonging to ray origin
	double b_x = params.f*g_x/(g_x-params.f);
	double b_y = params.f*g_y/(g_y-params.f);

	// calc the focal point belonging to the direction of the ray
	double test1=dot(params.orientation,ray.direction);
	double3 focalPoint=params.root+params.f*1/(abs(dot(params.orientation,ray.direction)))*ray.direction;

	// deflect ray such that it will hit the focal point
	if (coat_reflected)
		ray.direction=-normalize(focalPoint-ray.position);
	else
		ray.direction=normalize(focalPoint-ray.position);

	// now the wavefront of the outgoing ray has to be such that it becomes zero at the image point
	double2 radiusPrime_r;
	radiusPrime_r.x=b_x*abs(dot(ray.direction,hitParams.normal));
	radiusPrime_r.y=b_y*abs(dot(ray.direction,hitParams.normal));

	// if we are sitting in caustic here, we need to move the ray out of it
	if ( (b_x==0)||(b_y==0) )
	{
		radiusPrime_r.x=ray.lambda;
		radiusPrime_r.y=ray.lambda;
		ray.opl=ray.opl+ray.lambda;
		ray.position=ray.position+ray.lambda*ray.direction;
	}

	//refractDifferentialData(oldRayDirection, hitParams.normal, ray.direction,  P_r, radius_r, radiusBar_r, torsion_r, torsionBar_r, mu, PPrime_r, TPrime_r, radiusPrime_r, torsionPrime_r, ray.flux);	
	double3 PPrime_r=P_r;
	double3 TPrime_r=T_r;
	double torsionPrime_r=torsion_r;

	// transform the data of the refracted ray into its local system
	invTransformDifferentialData(PPrime_r, TPrime_r, radiusPrime_r, torsionPrime_r, ray.mainDirX, ray.mainDirY, ray.wavefrontRad);

	// apply OPL that corresponds to passage through ideal lense
	double dist2Root=calcDistRayPoint(params.root, params.orientation, ray.position);
	ray.opl=ray.opl+ray.nImmersed*(params.thickness-sqrt(pow(dist2Root,2)+pow(params.f,2))); // the OPL is calculated such that it will be equal for all rays meeting in the focal point


	return true;
};

#endif


