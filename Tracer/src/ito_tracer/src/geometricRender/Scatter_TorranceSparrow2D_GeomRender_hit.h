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

/**\file Scatter_TorranceSparrow1D_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef SCATTER_TORRANCESPARROW2D_GEOMRENDER_HIT_H
  #define SCATTER_TORRANCESPARROW2D_GEOMRENDER_HIT_H
  
#include "../Scatter_TorranceSparrow2D_hit.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

/**
 * \detail hitTorranceSparrow2D_PathTrace 
 *
 * modifies the raydata according to the parameters of the scatter
 *
 * \param[in] rayStruct_PathTracing &prd, Mat_hitParams hitParams, ScatTorranceSparrow2D_PathTrace_params params
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitTorranceSparrow2D_PathTrace(rayStruct_PathTracing &prd, Mat_hitParams hitParams, ScatTorranceSparrow2D_params params)
{

	uint32_t x1[5];
	ScatTorranceSparrow2D_params test=params;
	RandomInit(prd.currentSeed, x1); // init random variable
	double angleRay2Normal=acos(dot(prd.direction,-hitParams.normal)); // calc angle of ray to normal
	double3 rotAxis=cross(params.scatAxis,hitParams.normal); // calc axis around which these vectors are rotated to each other
	double3 rotAxis2=cross(rotAxis,params.scatAxis); // calc 2nd rotation axis

	if (params.impAreaType==AT_INFTY)
	{
		// calc scatter angle
		double xi=RandomGauss(x1); // get normal distribution with zero mean and unit variance
		double sqrtpi=sqrt(PI);
		double BRDFmax=params.Kdl*2+sqrtpi*(params.Ksl*params.Ksl+params.Ksp*params.Ksp);
		double ii=Random(x1)*BRDFmax; // get uniform distribution between zero and maximum BRDF to decide which mean and variance we need for our gaussian distribution
		double sigma=params.sigmaXsp; // init sigma to sigma of specular peak
		if (ii>sqrtpi*params.Ksp*params.Ksp)
		{
			prd.depth++;
			sigma=params.sigmaXsl; // set sigma to sigma of specular lobe
		}
		xi=xi*sigma;
		if (ii>sqrtpi*params.Ksp+params.Ksl*params.Ksl)
		{
			// the diffuse lobe distributes the direction uniformly
			xi=(Random(x1)-0.5)*PI; // get uniform deviate between -pi/2 pi/2
			//xi=asin(xi);// if we end up in the diffuse lobe we calculate xi from the inversion mehtod. see W.Press, Numerical Recipes 2nd ed., pp. 291 for reference
			prd.depth++;
		}
		prd.currentSeed=x1[4]; // save new seed for next randomization

		// rotate ray according to scattering angle
		rotateRay(prd.direction, rotAxis, xi);

		// do the same thing again for the 2nd ray component
		// calc scatter angle
		xi=RandomGauss(x1); // get normal distribution with zero mean and unit variance
		ii=Random(x1)*BRDFmax; // get uniform distribution between zero and maximum BRDF to decide which mean and variance we need for our gaussian distribution
		sigma=params.sigmaXsp; // init sigma to sigma of specular peak
		if (ii>sqrtpi*params.Ksp*params.Ksp)
		{
			prd.depth++;
			sigma=params.sigmaXsl; // set sigma to sigma of specular lobe
		}
		xi=xi*sigma;
		if (ii>sqrtpi*params.Ksp+params.Ksl*params.Ksl)
		{
			// the diffuse lobe distributes the direction uniformly
			xi=(Random(x1)-0.5)*PI; // get uniform deviate between -pi/2 pi/2
			//xi=asin(xi);// if we end up in the diffuse lobe we calculate xi from the inversion mehtod. see W.Press, Numerical Recipes 2nd ed., pp. 291 for reference
			prd.depth++;
		}
		prd.currentSeed=x1[4]; // save new seed for next randomization

		// rotate 2nd ray component according to scattering angle
		rotateRay(prd.direction, rotAxis2, xi);
	}
	else
	{
		double3 oldDirection=prd.direction;
		// distribute rays uniformly inside importance area
		aimRayTowardsImpArea(prd.direction, prd.position, params.impAreaRoot, params.impAreaHalfWidth, params.impAreaTilt, params.impAreaType, prd.currentSeed);
		// adjust flux according ot probability of given scatter direction
		double angle=abs(acos(dot(prd.direction, oldDirection)));
		prd.flux=prd.flux*params.Kdl*cos(angle)+params.Ksl*exp(-(angle*angle)/(2*params.sigmaXsl))+params.Ksp*exp(-(angle*angle)/(2*params.sigmaXsp));
	}

	return true;
};

#endif
