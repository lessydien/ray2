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

/**\file ParabolicSurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef PARABOLICSURFACEINTERSECT_H
  #define PARABOLICSURFACEINTERSECT_H
  
/* include geometry lib */
#include "Geometry_intersect.h"
#include "rayTracingMath.h"
#include <optixu/optixu_aabb.h>

/* declare class */
/**
  *\class   ParabolicSurface_ReducedParams
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
class ParabolicSurface_ReducedParams : public Geometry_ReducedParams
{
	public:
 	  double3 centre;
	  double3 orientation;
	  double2 curvatureRadius;
	  double2 apertureRadius;
	  double3 root;
//	  double rotNormal; // rotation of geometry around its normal
	  ApertureType apertureType;
	  double3 normal;
	  //int geometryID;
};

/**
 * \detail intersectRaySphere 
 *
 * calculates the intersection of a ray with a sinus normal surface
 *
 * \param[in] double3 rayPosition, double3 rayDirection, ParabolicSurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */






inline RT_HOSTDEVICE double intersectRayParabol(double3 rayPosition, double3 rayDirection, ParabolicSurface_ReducedParams params)
{
	// calc the two intersection points


	// Drehung -> Implentieren der Drehfunktion!

	double3x3 Mx=make_double3x3(1,0,0, 0,cos(-params.tilt.x),-sin(-params.tilt.x), 0,sin(-params.tilt.x),cos(-params.tilt.x));
	double3x3 My=make_double3x3(cos(-params.tilt.y),0,sin(-params.tilt.y), 0,1,0, -sin(-params.tilt.y),0,cos(-params.tilt.y));
	double3x3 Mz=make_double3x3(cos(-params.tilt.z),-sin(-params.tilt.z),0, sin(-params.tilt.z),cos(-params.tilt.z),0, 0,0,1);
	double3x3 Mxy=Mx*My;
	double3x3 M=Mxy*Mz;

	double3x3 Mzy = Mz*My;
	double3x3 Mzyx = Mzy * Mx;

	
	// Alles ins System des Paraboloiden drehen, dafür benöitigt: gedrehter Teststrahl + Testrichtung und Versatz des Paraboloiden zum Nullpunkt

	double3 shift = Mzyx* params.centre;
	double3 rayDirection_rot = Mzyx * rayDirection;
	double3 rayPosition_rot = Mzyx * rayPosition - shift;


	// Quadratische Lösungsformel
	
	double a = rayDirection_rot.x*rayDirection_rot.x + rayDirection_rot.y*rayDirection_rot.y;
	double b = 2* rayPosition_rot.x * rayDirection_rot.x + 2* rayPosition_rot.y * rayDirection_rot.y - params.curvatureRadius.x * rayDirection_rot.z;
	double c = rayPosition_rot.x * rayPosition_rot.x + rayPosition_rot.y * rayPosition_rot.y - params.curvatureRadius.x * rayPosition_rot.z;



	double rt = b*b - 4 * a * c;
	 if (rt<0)
	{
	  return 0;
	}

	 // Falls a == 0 gibt es nur einen Schnittpunkt, Lösungsformel funzt nicht

	 if (a==0)
	 {
	double t = (rayPosition_rot.x * rayPosition_rot.x + rayPosition_rot.y * rayPosition_rot.y) / (params.curvatureRadius.x * rayDirection_rot.z) - rayPosition_rot.z  / rayDirection_rot.z;
	double3 intersection=rayPosition+t*rayDirection;
	double3 intersection_rot=rayPosition_rot+t*rayDirection_rot;
	
	if (checkAperture(params.centre, params.tilt, intersection, params.apertureType, params.apertureRadius))
	{
		return t;
	}
	else
	{
		return 0;
	}
	 }

	 // Streckungsfaktoren für beide Lösungen der quadratischen Gleichung berechnen

	double t1 = (-b + sqrt(rt)) / (2*a);
	double t2 = (-b - sqrt(rt)) / (2*a);

	double3 intersection1=rayPosition+t1*rayDirection;
	double3 intersection2=rayPosition+t2*rayDirection;

	double3 intersection1_rot=rayPosition_rot+t1*rayDirection_rot;
	double3 intersection2_rot=rayPosition_rot+t2*rayDirection_rot;


	//Überprüfen, welche Lösung die Richtige ist und als t ausgeben

	if ( abs(t1) <= abs(t2) && checkAperture(params.centre, params.tilt, intersection1, params.apertureType, params.apertureRadius) && t1 >= 0)
	{
		double t = t1;
		double3 intersection=rayPosition+t*rayDirection;
		return t;
	}
	else if ( abs(t2) < abs(t1) && checkAperture(params.centre, params.tilt, intersection2, params.apertureType, params.apertureRadius) && t2 >= 0)
	{
		double t = t2;
		double3 intersection=rayPosition+t*rayDirection;
		return t;
	}
	else
	{
	
		if ( abs(t1) <= abs(t2) && checkAperture(params.centre, params.tilt, intersection2, params.apertureType, params.apertureRadius) && t2 >= 0)
		{
		double t = t2;
		double3 intersection=rayPosition+t*rayDirection;
		return t;
		}
		else if ( abs(t2) < abs(t1) && checkAperture(params.centre, params.tilt, intersection1, params.apertureType, params.apertureRadius) && t1 >= 0)
		{
			double t = t1;
			double3 intersection=rayPosition+t*rayDirection;
			return t;
		}
		else
		{
			return 0;
		}
	}

}
	
	


/**
 * \detail calcHitParamsSphere 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,ParabolicSurface_ReducedParams params
 * 
 * \return Mat_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsParabol(double3 position, ParabolicSurface_ReducedParams params)
{
	double3 n;

	// ins System des Paraboloiden drehen

	rotateRay(&position, -params.tilt);
	rotateRay(&params.centre, -params.tilt);
	
	n = make_double3(2* (position.x -  params.centre.x) / params.curvatureRadius.x, 2* (position.y - params.centre.y) / params.curvatureRadius.x, -1);
	
	// Normalenvektor ins globale System zurückdrehen

	rotateRay(&n, params.tilt);	
	
	Mat_hitParams t_hitParams;
	t_hitParams.normal=normalize(n);
	return t_hitParams;
}

/**
 * \detail parabolicSurfaceBounds 
 *
 * calculates the bounding box of a parabolicSurface
 *
 * \param[in] int primIdx, float result[6], ParabolicSurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE void parabolicSurfaceBounds (int primIdx, float result[6], ParabolicSurface_ReducedParams params)
{
    optix::Aabb* aabb = (optix::Aabb*)result;  
    aabb->invalidate();
}

#endif
