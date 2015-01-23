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

/**\file MaterialRefracting_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MATERIALREFRACTING_DIFFRAYS_HIT_H
#define MATERIALREFRACTING_DIFFRAYS_HIT_H

#include "../rayTracingMath.h"
#include "Material_DiffRays_hit.h"
#include "../MaterialRefracting_hit.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

/* declare class */
/**
  *\class   MatRefracting_DiffRays_params 
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
class MatRefracting_DiffRays_params : public MatRefracting_params
{
public:
	//double n1; // refractive index1
	//double n2; // refractive index2
	////double t; // amplitude transmission coefficient
	////double r; // amplitude reflection coefficient
};

/**
 * \detail hitRefracting_DiffRays 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in] diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, MatRefracting_params params, double t_hit, int geomID, bool coat_reflected
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitRefracting_DiffRays(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, MatRefracting_params params, double t_hit, int geomID, bool coat_reflected)
{
	MatRefracting_params test=params;
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
	//double3 Pr=cross(ray.direction,normal);
	//if (length(Pr)==0)
	//	Pr=mainDirX; // if direction and normal are parallel, we can arbitrarily chose one of the mainDirections as Pr. ( can we really??? )
	//Pr=normalize(Pr);
	//double3 Tr=normalize(cross(ray.direction,Pr));
	//double cosPhi=dot(Pr, ray.mainDirX);
	//double sinPhi=-dot(Tr, ray.mainDirX);

	//double curvXr=cosPhi*cosPhi/ray.wavefrontRad.x+sinPhi*sinPhi/ray.wavefrontRad.y;

	//double curvYr=sinPhi*sinPhi/ray.wavefrontRad.x+cosPhi*cosPhi/ray.wavefrontRad.y;
	//double torsion=(1/ray.wavefrontRad.x-1/ray.wavefrontRad.y)*sinPhi*cosPhi;

	// transform the data of the surface
	double3 PBar_r, TBar_r;
	double2 radiusBar_r;
	double torsionBar_r;
	transformDifferentialData(ray.direction, hitParams.mainDirY, hitParams.mainRad, hitParams.normal, PBar_r, TBar_r, radiusBar_r, torsionBar_r );
	//double3 PrBar=Pr;
	//double3 TrBar=normalize(cross(normal,PrBar));
	//double cosPhiBar=dot(PrBar,mainDirX);
	//double sinPhiBar=-dot(TrBar,mainDirX);
	//double curvXrBar=cosPhiBar*cosPhiBar/mainRad.x+sinPhiBar*sinPhiBar/mainRad.y;
	//double curvYrBar=sinPhiBar*sinPhiBar/mainRad.x+cosPhiBar*cosPhiBar/mainRad.y;
	//double torsionBar=(1/mainRad.x-1/mainRad.y)*sinPhiBar*cosPhiBar;

	//double mu, gamma;
	//double s=dot(ray.direction,normal);
	//int signS=(int)(s/abs(s));
	// if the coating wants us to have reflection, we do reflection here instead of refraction
	double mu;
	double3 PPrime_r, TPrime_r;
	double2 radiusPrime_r;
	double torsionPrime_r;
	// do the refraction
	if (coat_reflected)
	{
		ray.direction=reflect(ray.direction,hitParams.normal);
		mu=1;
		// calc the data of the reflected ray
		reflectDifferentialData(oldRayDirection, hitParams.normal, ray.direction,  P_r, radius_r, radiusBar_r, torsion_r, torsionBar_r, mu, PPrime_r, TPrime_r, radiusPrime_r, torsionPrime_r, ray.flux);

//		gamma=-2*s;
	}
	else
	{
		if (ray.nImmersed==params.n1)
		{
			mu=ray.nImmersed/params.n2;
			if (calcSnellsLaw(&(ray.direction),hitParams.normal, ray.nImmersed, params.n2))
				ray.nImmersed=params.n2;
		}
		else if (ray.nImmersed==params.n2)
		{
			mu=ray.nImmersed/params.n1;
			if (calcSnellsLaw(&(ray.direction),hitParams.normal, ray.nImmersed, params.n1))
				ray.nImmersed=params.n1;
		}
		else
		{
			// some error mechanism
			return 0;
		}
		// calc the data of the refracted ray
		refractDifferentialData(oldRayDirection, hitParams.normal, ray.direction,  P_r, radius_r, radiusBar_r, torsion_r, torsionBar_r, mu, PPrime_r, TPrime_r, radiusPrime_r, torsionPrime_r, ray.flux);
//		gamma=-mu*s+signS*sqrt(1-mu*mu*(1-s*s));
	}
	//double3 PrPrime=Pr;
	//double3 TrPrime=normalize(cross(ray.direction,PrPrime));
	//double curvXrPrime=mu*curvXr+gamma*curvXrBar;
	//double sPrime=dot(ray.direction,normal);
	//double curvYrPrime=1/(sPrime*sPrime)*(mu*s*s*curvYr+gamma*curvYrBar);
	//double torsionPrime=1/sPrime*(mu*s*torsion+gamma*torsionBar);

	// transform the data of the refracted ray into its local system
	invTransformDifferentialData(PPrime_r, TPrime_r, radiusPrime_r, torsionPrime_r, ray.mainDirX, ray.mainDirY, ray.wavefrontRad);
	//double PhiPrime;
	//if (torsionPrime==0)
	//	PhiPrime=0;
	//else
	//{
	//	if (curvXrPrime==curvYrPrime)
	//		PhiPrime=torsionPrime/abs(torsionPrime)*PI/2;
	//	else
	//		PhiPrime=(atan(2*torsionPrime/(curvXrPrime-curvYrPrime))/2);
	//}
	//double cosPhiPrime=cos(PhiPrime);
	//double sinPhiPrime=sin(PhiPrime);
	//double curvXPrime=cosPhiPrime*cosPhiPrime*curvXrPrime+sinPhiPrime*sinPhiPrime*curvYrPrime-sin(2*PhiPrime)*torsionPrime;
	//double curvYPrime=sinPhiPrime*sinPhiPrime*curvXrPrime+cosPhiPrime*cosPhiPrime*curvYrPrime+sin(2*PhiPrime)*torsionPrime;
	//ray.wavefrontRad.x=1/curvXPrime;
	//ray.wavefrontRad.y=1/curvYPrime;
	//ray.mainDirX=cosPhiPrime*PrPrime-sinPhiPrime*TrPrime;
	//ray.mainDirY=sinPhiPrime*PrPrime+cosPhiPrime*TrPrime;

	// adjust flux due to projections on interface
	ray.flux=ray.flux*abs(dot(oldRayDirection,hitParams.normal))/abs(dot(ray.direction,hitParams.normal));

	return 1;
}

#endif


