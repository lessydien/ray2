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

/**\file ApertureArraySurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef APERTUREARRAYSURFACEINTERSECT_H
  #define APERTUREARRAYSURFACEINTERSECT_H
  
/* include geometry lib */
#include "Geometry_intersect.h"
#include "rayTracingMath.h"
#include "PlaneSurface_intersect.h"
//#include "SphericalSurface_intersect.h"


/* declare class */
/**
  *\class   ApertureArraySurface_ReducedParams
  *\ingroup Geometry
  *\brief   reduced set of params that is loaded onto the GPU if the tracing is done there
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
class ApertureArraySurface_ReducedParams : public Geometry_ReducedParams
{
	public:
 	  double3 root;
	  double3 normal;
	  double2 microAptPitch;
	  double2  microAptRad;
	  ApertureType microAptType;
	  ApertureType apertureType;
	  double2 apertureRadius;
};

/**
 * \detail intersectRaySphere 
 *
 * calculates the intersection of a ray with a the surface of a micro lens array
 *
 * \param[in] double3 rayPosition, double3 rayDirection, ApertureArraySurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayApertureArraySurface(double3 rayPosition, double3 rayDirection, ApertureArraySurface_ReducedParams params)
{

	double t = intersectRayPlane(rayPosition, rayDirection, params.root, params.normal);

	// check aperture
	if ( !checkAperture(params.root, params.tilt, rayPosition+t*rayDirection, params.apertureType, params.apertureRadius) )
	{
		return 0;
	}
	else
	{
		// position on micro lens array surface in local coordinate system 
		double3 tmpPos=rayPosition+t*rayDirection-params.root;
		rotateRayInv(&tmpPos,params.tilt);
		double3 tmpDir=rayDirection;
		rotateRayInv(&tmpDir,params.tilt);

		// see in which subaperture we are
		double fac=floor(tmpPos.x/params.microAptPitch.x+0.5);
		double3 microAptCentre;
		microAptCentre.x=fac*params.microAptPitch.x;
		fac=floor(tmpPos.y/params.microAptPitch.y+0.5);
		microAptCentre.y=fac*params.microAptPitch.y;
		microAptCentre.z=0;

		//**********************************************
		// check wether this intersection is inside the aperture
		//**********************************************
		if ( checkAperture(microAptCentre, make_double3(0,0,0), tmpPos, params.microAptType, params.microAptRad) )
			return 0;				
	}

	return t;
}

/**
 * \detail calcHitParamsApertureArray 
 *
 * calculates the intersection of a ray with a the surface of a aperture array
 *
 * \param[in] double3 rayPosition, double3 rayDirection, ApertureArraySurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsApertureArraySurface(double3 position,ApertureArraySurface_ReducedParams params)
{
	Mat_hitParams l_hitParams;
	l_hitParams.normal=params.normal;
	return l_hitParams;
}

/**
 * \detail apertureArraySurfaceBounds 
 *
 * calculates the bounding box of an aperture array surface
 *
 * \param[in] int primIdx, float result[6], ApertureStop_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */

inline RT_HOSTDEVICE void apertureArraySurfaceBounds (int primIdx, float result[6], ApertureArraySurface_ReducedParams params)
{
    // bounding box is the same as that of a plane surface
    PlaneSurface_ReducedParams planeParams;
    planeParams.root=params.root;
    planeParams.apertureRadius=params.apertureRadius;
    planeParams.tilt=params.tilt;
    planeSurfaceBounds(primIdx, result, planeParams);
}

#endif
