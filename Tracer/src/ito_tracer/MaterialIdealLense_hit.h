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

#ifndef MATERIALIDEALLENSE_HIT_H
#define MATERIALIDEALLENSE_HIT_H

#include "rayTracingMath.h"
#include "Material_hit.h"

/* declare class */
/**
  *\class   MatIdealLense_params 
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
class MatIdealLense_params
{
public:
	double f; //!> focal length of ideal lense
	double3 root; //!> root of the ideal lense
	double3 orientation; //!> orientation of the ideal lense
	double thickness; //!> thickness of ideal lense. This comes from the fact that the ray through the centre of the ideal lense hast to have a phase shift relative to the ray through the outermost corner of the aperture such that a perfect spherical wavefront appears behind the ideal lense
};

/**
 * \detail hitIdealLense 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] rayStruct &ray, Mat_hitParams hitParams, MatIdealLense_params params, double t_hit, int geomID, bool coat_reflected
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitIdealLense(rayStruct &ray, Mat_hitParams hitParams, MatIdealLense_params params, double t_hit, int geomID, bool coat_reflected)
{
	MatIdealLense_params test=params;
	ray.position=ray.position+t_hit*ray.direction;
	ray.currentGeometryID=geomID;
	// apply OPL up to intersection with ideal lense
	ray.opl=ray.opl+ray.nImmersed*t_hit;
	// apply OPL that corresponds to passage through ideal lense
	double dist2Root=calcDistRayPoint(params.root, params.orientation, ray.position);
	ray.opl=ray.opl+ray.nImmersed*(params.thickness-sqrt(pow(dist2Root,2)+pow(params.f,2))); // the OPL is calculated such that it will be equal for all rays meeting in the focal point
	// deflect ray such that it hits in the focal point belonging to the direction of the ray
	double test1=dot(params.orientation,ray.direction);
	double3 focalPoint=params.root+params.f*1/(abs(dot(params.orientation,ray.direction)))*ray.direction;
	if (coat_reflected)
		ray.direction=-normalize(focalPoint-ray.position);
	else
		ray.direction=normalize(focalPoint-ray.position);
	return true;
};

#endif


