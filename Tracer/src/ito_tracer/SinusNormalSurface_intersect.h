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

/**\file SinusNormalSurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef SINUSNORMALSURFACEINTERSECT_H
  #define SINUSNORMALSURFACEINTERSECT_H

/* include header of basis class */
#include "Geometry_intersect.h"
#include "rayTracingMath.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

/* declare class */
/**
  *\class   SinusNormalSurface_ReducedParams
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
class SinusNormalSurface_ReducedParams : public Geometry_ReducedParams
{
  public:
   double3 root;
   double3 normal;
   double2 apertureRadius;
//   double rotNormal; // rotation of geometry around its normal
   ApertureType apertureType;
   double period; //!> period of cosine profile
   double ampl; //!> amplitude of cosine profile
   double3 grooveAxis; //!> axis parallel to grooves
   double iterationAccuracy; //!> if the calculated intersection point is within this accuracy, we stop the iteration loop
   //int geometryID;
};

/**
 * \detail intersectRaySinusNormalSurface 
 *
 * calculates the intersection of a ray with a sinus normal surface
 *
 * \param[in] double3 rayPosition, double3 rayDirection, SinusNormalSurface_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRaySinusNormalSurface(double3 rayPosition, double3 rayDirection, SinusNormalSurface_ReducedParams params)
{
	double3 rayDir, rayPos, planeRoot, planeNormal;
	rayDir=rayDirection;
	rayPos=rayPosition;
	planeRoot=params.root;
	planeNormal=params.normal;
	SinusNormalSurface_ReducedParams paramsTest=params;

	double3 newRayPos=rayPosition;
	double t=0;
	// if we are already within the range of the surface peaks (else...), we start the iteration at the ray position. if not (if...) we start at the intersection of the ray with the plane containing the peaks of the surface pointing towards the ray
	if ( !(abs(intersectRayPlane(newRayPos, -params.normal, params.root, params.normal))<params.ampl) )
	{
		// check wether we are in front of the equlibrium surface or behind it/ positive sign means we are behind the plane ( in negative normal direction )
		float signPf=dot(params.normal,params.root-rayPosition)/length(params.root-rayPosition);
		int signP=(int)(signPf/abs(signPf)); //

		// calc intersection of ray with plane containing the peaks of cos surface
		t = intersectRayPlane(rayPosition, rayDirection, params.root-signP*params.ampl*params.normal, params.normal);
	}
	else
	{
		// if we start from inside the profile, we first check wether we will hit the profile again
		// if the old position is within the surface profile, we need to move the ray far enough so we are outside the iteration accurcay to start the iteration...
		t=params.period/2/abs(dot(rayDirection,params.grooveAxis));
	}
	newRayPos=rayPosition+t*rayDirection;

	// if we don't hit the equilibrium plane, we don't hit the surface at all...
	if (t<0)
		return 0;
	// we calculate the distance of the ray to the surface (ft) and iterate for roots of this function...
	// iterate until we are closer than 10nm to the surface
	double ftOld=DOUBLE_MAX;
	double delta_t;
	unsigned int count=0;
	for (;;)//(unsigned int count=0;count<50;count++) 
	{
		double k=2*PI/params.period;
		// calc distance of ray to equilibrium surface / as we're calculating the distance from ray pos to eq. plane, we need to take the negative normal here!!!
		//double ft1=intersectRayPlane(newRayPos, -params.normal, params.root, params.normal);
		double ft1=dot(newRayPos-params.root,params.normal);
		// calc distance of cosine surface to equilibrium surface at ray position
//		double ft2=params.ampl*cos(k*dot(params.root-newRayPos,params.grooveAxis));
		double ft2=params.ampl*cos(k*dot(newRayPos-params.root,params.grooveAxis));
		// calc difference of ray position to respective position on surface
		double ft=ft2-ft1;
		if (abs(ft)<params.iterationAccuracy) // if we are close than say 10nm, we end iteration
			break;
		// if new ft is bigger than old but we didn't encountered a zero crossing, we're likely to just have passed a local minimum
		// Therefore we jump ahead half a period to be sure to come into the valley of the next minimum
		if ( (abs(ft)>abs(ftOld)) && (ft*ftOld>0) )
		{
			delta_t=0.75*params.period/abs(dot(rayDirection,params.grooveAxis));
			ftOld=DOUBLE_MAX; // reset ftOld
			//delta_t=-abs(ft/ftOld)*delta_t; // if we got worse in the last step or changed sign, we go back a fraction of the last step...
			//ft=ftOld; // reuse old ft
		}
		else
		{
			// save ft for next iteration
			ftOld=ft;

			// calc derivative of distance to surface with respect to t
			double dt3=k*dot(rayDirection,params.grooveAxis);
			double dt2=-params.ampl*sin(k*dot(newRayPos-params.root,params.grooveAxis));
			double dt1=dot(rayDirection,params.normal);
//			double dt1=dot(rayDirection,params.normal); 
//			double dt21=params.ampl*cos(k*dot(params.root-rayPosition,params.grooveAxis))*sin(-k*t*dot(params.grooveAxis,rayDirection))*k*dot(params.grooveAxis,rayDirection);
//			double dt22=params.ampl*sin(k*dot(params.root-rayPosition,params.grooveAxis))*cos(-k*t*dot(params.grooveAxis,rayDirection))*k*dot(params.grooveAxis,rayDirection);
			/***********************************************
			/
			// it might be possible to come to a point, where we are right in the middle of two roots and dt=0. How do we handle this ???
			/
			/***********************************************/
			double dt;
//			if (dt1*dt2*dt3>0) // if the derivatives of the ray and the surface point in the same direction, we use the ray derivative only
//				dt=dt1;
//			else
				dt=dt1-dt3*dt2;
//			double dt=dt1-dt21+dt22; 
			delta_t=ft/dt;
			if (abs(delta_t)>abs(0.2*params.period/dot(rayDirection,params.grooveAxis)))
				delta_t=delta_t/abs(delta_t)*0.25*params.period/abs(dot(rayDirection,params.grooveAxis));
		}
		// calc new t
		t=t+delta_t;
		// update ray position for next iteration
		newRayPos=rayPosition+t*rayDirection;
		count++;
		// exit if we don't converge
		if (count>50)
			return 0;
	}
//	double3 test=rayPosition+t*rayDirection;
//	SinusNormalSurface_Params testPar=params;
	// check aperture
	if ( !checkAperture(params.root, params.tilt, rayPosition+t*rayDirection, params.apertureType, params.apertureRadius) )
	{
		return 0;
	}
	return t;
}

/**
 * \detail calcHitParamsSinusNormalSurface 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,SinusNormalSurface_ReducedParams params
 * 
 * \return Mat_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_hitParams calcHitParamsSinusNormalSurface(double3 position,SinusNormalSurface_ReducedParams params)
{
	double3 Pos=position;
	SinusNormalSurface_ReducedParams testPar=params;
	// calc angle of rotation
	double alpha=atan(params.ampl*2*PI/params.period*sin(2*PI/params.period*dot(params.root-position,params.grooveAxis)));
	// init normal to surface normal
	double3 n=params.normal;
	// rotate normal around axis parallel to grooves
	double3 rotAxis=cross(params.normal,params.grooveAxis);
	rotateRay(n,rotAxis,-alpha);
	Mat_hitParams t_hitParams;
	t_hitParams.normal=n;
	return t_hitParams;
}

#endif
