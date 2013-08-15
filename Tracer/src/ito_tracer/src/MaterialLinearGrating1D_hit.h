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

/**\file MaterialLinearGrating1D_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef MATERIALLINEARGRATING1D_HIT_H
#define MATERIALLINEARGRATING1D_HIT_H

#include "randomGenerator.h"
#include "rayTracingMath.h"
#include "Material_hit.h"
#include <time.h>

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif


/* declare class */
/**
  *\class   MatLinearGrating1D_params 
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
class MatLinearGrating1D_params
{
public:
	short nrDiffOrders; // number of diffraction orders with non zereo efficiency
	double g; // grating constant ( period of grating )
	double eff[MAX_NR_DIFFORDERS]; // efficiencies of diffraction orders starting with maximum efficiency
	short diffOrderNr[MAX_NR_DIFFORDERS]; // number of diffraction order that corresponds to respective efficiency
	double3 diffAxis; // axis along which diffraction arises. I.e. the axis perpendicular to that axis along which the 1D grating is homogeneous...
	double nRefr1; // refractive index inside element. 0 indicates reflective grating
	double nRefr2; // refractive index outside element
};

/**
 * \detail hitLinearGrating1D 
 *
 * modifies the raydata according to the parameters of the material
 *
 * \param[in]rayStruct &ray, Mat_hitParams hitParams, MatLinearGrating1D_params params, double t_hit, int geomID, bool reflected
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitLinearGrating1D(rayStruct &ray, Mat_hitParams hitParams, MatLinearGrating1D_params params, double t_hit, int geomID, bool reflected)
{
	ray.position=ray.position+t_hit*ray.direction;
	ray.currentGeometryID=geomID;
	ray.opl=ray.opl+ray.nImmersed*t_hit;
	uint32_t x1[5];
	int xi=0;
	MatLinearGrating1D_params paramsTest=params;
	double effSum=0;
	double maxEff=0;
	short maxEffIndex=0;
	short i;
	for (i=0;i<params.nrDiffOrders;i++)
	{
		effSum=effSum+params.eff[i];
		if (params.eff[i]>maxEff)
		{
			maxEff=params.eff[i];
			maxEffIndex=i;
		}
	}

	int count=0;
	if (params.nrDiffOrders!=1)
	{
		RandomInit(ray.currentSeed, x1); // seed random generator
		/* chose diffraction order to reflect into */
		// see http://web.physik.rwth-aachen.de/~roth/eteilchen/MonteCarlo.pdf slide 12 for reference. we need a better reference !!!
		double fxi=0; // init 
		double yi=1; // init
		//int count=0;
		for (count=0;count<2000;count++)//while (yi>=fxi) // this could turn into an endless loop !! maybe we should implement some break backup
		{
			// calc uniformly distributed number
			xi=IRandom(0, params.nrDiffOrders-1, x1); // random between 0,1
			// calc uniformly distributed number between 0 and max of scattering function
			yi=Random(x1); // random between 0,1
			yi=yi*maxEff; // random between 0 and maximum diffraction efficiency
			// look up probability of xi from diffraction efficiencies
			fxi=params.eff[xi];
			if (yi<=fxi)
				break;
		}
		ray.currentSeed=x1[4]; // save seed for next round
	}
	double mu;
	if (params.nRefr1==0)
		mu=1; // if we have a reflective grating ...
	else
	{
		if (ray.nImmersed==params.nRefr1)
			mu=ray.nImmersed/params.nRefr2;
		else
			mu=ray.nImmersed/params.nRefr1;
	}
	/* see JOSA Vol. 52 pp. 672, Spencer, Murty, General Ray-Tracing Procedure */
	double p=2*mu*dot(ray.direction,hitParams.normal);
	double Delta=params.diffOrderNr[xi]*ray.lambda/params.g;
	double q=(mu*mu-1+Delta*Delta-2*mu*Delta*dot(ray.direction,params.diffAxis));
	double tau, tau1, tau2;
	tau1=-p/2+sqrt(p*p/4-q);
	tau2=-p/2-sqrt(p*p/4-q);
	// for a reflecting grating we need the tau that has the bigger magnitude
	if (reflected || (params.nRefr1==0) ) // if we have reflection grating
		if (abs(tau1)>abs(tau2))
			tau=tau1;
		else
			tau=tau2;
	else // if we have transmission grating
		if (abs(tau1)>abs(tau2))
			tau=tau2;
		else
			tau=tau1;
	// update ray direction
	ray.direction=mu*ray.direction-Delta*params.diffAxis+tau*hitParams.normal;
//	ray.direction=normalize(ray.direction);
	// if we diffracted in another diffraction order than the on with the highest efficiency, we increment depth
	if (xi != maxEffIndex)
		ray.depth++;
	// set flux to maximum efficiency. This way we don't have to throw away most of the rays. The further variation of the flux dependent on reflection angle is modelled via the number of rays generated in each direction... 
	ray.flux=effSum*ray.flux; 
	return true;
}

#endif


