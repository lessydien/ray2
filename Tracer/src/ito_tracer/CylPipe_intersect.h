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

/**\file CylPipe_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef CYLPIPEINTERSECT_H
  #define CYLPIPEINTERSECT_H

#include "Geometry_intersect.h"
#include "rayTracingMath.h"

/* declare class */
/**
  *\class   CylPipe_ReducedParams
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
class CylPipe_ReducedParams : public Geometry_ReducedParams
{
  public:
 	  double3 root;
	  double3 orientation;
	  double2 radius;
//	  double rotNormal; // rotation of geometry around its normal
	  double thickness;   
	  //int geometryID;
};

/**
 * \detail intersectRayCylPipe 
 *
 * calculates the intersection of a ray with an cylindrical pipe
 *
 * \param[in] double3 rayPosition, double3 rayDirection, CylPipe_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayCylPipe(double3 rayPosition, double3 rayDirection, CylPipe_ReducedParams params)
{
	/* see "http://www.matheboard.de/archive/1188/thread.html (20.07.10)" for a derivation of the algorithm */
	/* "P.J. Schneider, Geometric Tools for Computer Graphics, pp512" might be a more serious source... */
	double3 e=rayDirection-dot(rayDirection,params.orientation)/dot(params.orientation,params.orientation)*params.orientation;
	double3 f=(rayPosition-params.root)-dot((rayPosition-params.root),params.orientation)/dot(params.orientation,params.orientation)*params.orientation;
  
	double A=dot(e,e);
	double B=2*dot(e,f); 
	double C=dot(f,f)-params.radius.x*params.radius.x;
	double root=(B*B-4*A*C);
	
	double3 intersect;		//intersection Point		

	// if root is positive we have two intersections if not, we have none
	if (root>=0)
	{
		// calc the nearest intersection

		//double t=2*C/(-B+sqrt(root));
		double t1=(-B+sqrt(root))/(2*A);
		double t2=(-B-sqrt(root))/(2*A);

		if ((t1<t2)&&(t1>0))
		{
			intersect=rayPosition+t1*rayDirection;
			if (checkApertureCylinder(params.root,params.orientation,intersect,params.thickness/2))
			{
				return t1;
			}
		}
		//if we didn't return yet
		if (t2>0)
		{
			intersect=rayPosition+t2*rayDirection;
			if (checkApertureCylinder(params.root,params.orientation,intersect,params.thickness/2))
			{
				return t2;
			}
		}
	}
	// if we haven't returned yet, there's no intersection which we indicate by returning zero
	return 0;
}

/**
 * \detail calcHitParamsCylPipe 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,CylPipe_ReducedParams params
 * 
 * \return Mat_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsCylPipe(double3 position,CylPipe_ReducedParams params)
{
	// first calculate the intersection of the middle axis of the cylinder with the plane through position beeing normal to the middle axis
	double t=intersectRayPlane(params.root, params.orientation, position, params.orientation);
	double3 i=params.root+t*params.orientation;
	// the normal vector we looked for is the normalized version of the vector connecting position and i
	Mat_hitParams t_hitParams;
	t_hitParams.normal=normalize(i-position);
	return t_hitParams;
}

#endif
