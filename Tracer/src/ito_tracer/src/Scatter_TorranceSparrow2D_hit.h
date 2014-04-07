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

#ifndef SCATTER_TORRANCESPARROW2D_HIT_H
  #define SCATTER_TORRANCESPARROW2D_HIT_H
  
#include "randomGenerator.h"
#include "rayTracingMath.h"
#include "rayData.h"
#include "Scatter_hit.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

/* declare class */
/**
  *\class   ScatTorranceSparrow2D_params
  *\ingroup Scatter
  *\brief   reduced set of params that are loaded onto the GPU if the tracing is done there
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
class ScatTorranceSparrow2D_params: public Scatter_ReducedParams
{
public:
	double Kdl; // coefficient of diffuse lobe
	double Ksl; // coefficient of specular lobe
	double Ksp; // coefficient of specular peak
	double sigmaXsl; // width parameter of specular lobe
	double sigmaXsp; // width parameter of specular peak
	double3 scatAxis;
};

/**
 * \detail hitTorranceSparrow2D 
 *
 * modifies the raydata according to the parameters of the scatter
 *
 * \param[in] rayStruct &prd, Mat_hitParams hitParams, ScatTorranceSparrow2D_params params
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitTorranceSparrow2D(rayStruct &prd, Mat_hitParams hitParams, ScatTorranceSparrow2D_params params)
{
	uint32_t x1[5];
	ScatTorranceSparrow2D_params test=params;
	RandomInit(prd.currentSeed, x1); // init random variable
	double angleRay2Normal=acos(dot(prd.direction,-hitParams.normal)); // calc angle of ray to normal
    double3 rotAxis=cross(params.scatAxis, hitParams.normal);

	if (params.impAreaType==AT_INFTY)
	{
		// calc scatter angle
		double xi=RandomGauss(x1); // get normal distribution with zero mean and unit variance
		double sqrtpi=sqrt(PI);
        double ii=Random(x1);
        double phi=0;
        double phi2=0;

        phi=Random(x1)*PI-PI/2;
        phi2=Random(x1)*PI-PI/2;
        if (ii>params.Kdl)
        {
            phi=RandomGauss(x1)*params.sigmaXsl; // set sigma to sigma of specular lobe
            phi2=RandomGauss(x1)*params.sigmaXsl;
        }
        if (ii>params.Kdl+params.Ksl)
        {
            phi=RandomGauss(x1)*params.sigmaXsp;
            phi2=RandomGauss(x1)*params.sigmaXsp;
        }
        prd.currentSeed=x1[4]; // save new seed for next randomization
        
		// rotate ray according to first scattering angle
		rotateRay(prd.direction, rotAxis, phi);
        // rotate scat axis according to first rotation
        double3 rotAxis2=params.scatAxis;
        rotateRay(rotAxis2, rotAxis, phi);
        // rotate ray according to second scattering angle
        rotateRay(prd.direction, rotAxis2, phi2);
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
