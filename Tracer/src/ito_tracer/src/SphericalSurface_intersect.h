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

/**\file SphericalSurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef SPHERICALSURFACEINTERSECT_H
  #define SPHERICALSURFACEINTERSECT_H
  
/* include geometry lib */
#include "Geometry_intersect.h"
#include "rayTracingMath.h"

/* declare class */
/**
  *\class   SphericalSurface_ReducedParams
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
class SphericalSurface_ReducedParams : public Geometry_ReducedParams
{
	public:
 	  double3 centre;
	  double3 orientation;
	  double2 curvatureRadius;
	  double2 apertureRadius;
//	  double rotNormal; // rotation of geometry around its normal
	  ApertureType apertureType;
	  //int geometryID;
};

/**
 * \detail intersectRaySphere 
 *
 * calculates the intersection of a ray with a sinus normal surface
 *
 * \param[in] double3 rayPosition, double3 rayDirection, SphericalSurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRaySphere(double3 rayPosition, double3 rayDirection, SphericalSurface_ReducedParams params)
{
  // see http://en.wikipedia.org/wiki/Line–sphere_intersection or David H. Eberly (2006), 3D game engine design: a practical approach to real-time computer graphics, 2nd edition, Morgan Kaufmann for reference
  //double A=dot(rayDirection, rayDirection); equals unity...
  double3 origin=rayPosition-params.centre;
  double B=dot(rayDirection,origin);
  double C=dot(origin,origin)-(params.curvatureRadius.x*params.curvatureRadius.x);
  double root=(B*B-C);
  // if root is positive we have two intersections if not, we have none
  if (root<=0)
  {
	  return 0;
  }
  // calc first intersection
  double t1=-B+sqrt(root);
  t1 = t1 > 0 ? t1 : 0;
  double t2=-B-sqrt(root);
  t2 = t2 > 0 ? t2 : 0;
  double t;
  t = t1<t2 ? t1 : t2;
  double3 intersection=rayPosition+t*rayDirection;
  // we use the zemax sign convention, i.e. a negative radius represents convex surfaces !!
  // if the product of the orientation vector, the vector connecting the centre of the sphere and the radius is negative we have the physical intersection. Otherwise we have to calculate the other intersection
  if ( (dot(params.orientation,(intersection-params.centre))*params.curvatureRadius.x>0) || (abs(t) < 1E-10) ) // we could already sit on the sphere. Then t=0 and we need to consider the other intersection
  {
	// use the other intersection
  	t = t==t1 ? t2 : t1;
	// check the new intersection
	intersection=rayPosition+t*rayDirection;
	if ( (dot(params.orientation,(intersection-params.centre))*params.curvatureRadius.x>0) || (abs(t) < 1E-10))
		return 0;
  }
  //**********************************************
  // check wether this intersection is inside the aperture
  //**********************************************
  intersection=rayPosition+t*rayDirection;

  if (checkAperture(params.centre, params.tilt, intersection, params.apertureType, params.apertureRadius) )
  {
    return t;
  }
  return 0;
}

/**
 * \detail calcHitParamsSphere 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,SphericalSurface_ReducedParams params
 * 
 * \return Mat_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsSphere(double3 position,SphericalSurface_ReducedParams params)
{
	double3 n;
	n=position - params.centre;
	Mat_hitParams t_hitParams;
	t_hitParams.normal=normalize(n);
	return t_hitParams;
}

#endif
