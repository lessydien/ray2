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

/**\file ApertureStop_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef APERTURESTOPINTERSECT_H
  #define APERTURESTOPINTERSECT_H

/* include header of basis class */
#include "Geometry_intersect.h"
#include "rayTracingMath.h"
#include "PlaneSurface_intersect.h"

/* declare class */
/**
  *\class   ApertureStop_ReducedParams 
  *\ingroup Geometry
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
class ApertureStop_ReducedParams : public Geometry_ReducedParams
{
  public:
   double3 root;
   double3 normal;
   double2 apertureRadius;
   double2 apertureStopRadius;
//   double rotNormal; // rotation of geometry around its normal
   ApertureType apertureType;
   //int geometryID;
};

/**
 * \detail intersectRayApertureStop 
 *
 * calculates the intersection of a ray with an aperture stop
 *
 * \param[in] double3 rayPosition, double3 rayDirection, ApertureStop_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayApertureStop(double3 rayPosition, double3 rayDirection, ApertureStop_ReducedParams params)
{
	ApertureStop_ReducedParams test;
	test=params;
	double3 rayDir, rayPos, planeRoot, planeNormal;
	rayDir=rayDirection;
	rayPos=rayPosition;
	planeRoot=params.root;
	planeNormal=params.normal;

	double t = intersectRayPlane(rayPosition, rayDirection, params.root, params.normal);

//	double3 test=rayPosition+t*rayDirection;
//	ApertureStop_Params testPar=params;
	// check aperture
	if ( !checkAperture(params.root, params.tilt, rayPosition+t*rayDirection, params.apertureType, params.apertureRadius) )
	{
		return 0;
	}
	else
	{
		// if we hit the inner aperture of the aperture stop, the ray passes the aperture
		if ( checkAperture(params.root, params.tilt, rayPosition+t*rayDirection, params.apertureType, params.apertureStopRadius) )
		{
			return 0;
		}
	}

	return t;
}

/**
 * \detail apertureArraySurfaceBounds 
 *
 * calculates the bounding box of an aperture stop
 *
 * \param[in] int primIdx, float result[6], ApertureStop_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE void apertureStopBounds (int primIdx, float result[6], ApertureStop_ReducedParams params)
{
    // bounding box is the same as that of a plane surface
    PlaneSurface_ReducedParams planeParams;
    planeParams.root=params.root;
    planeParams.apertureRadius=params.apertureRadius;
    planeParams.tilt=params.tilt;
    planeSurfaceBounds(primIdx, result, planeParams);
}


#endif
