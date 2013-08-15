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

/**\file Coating_DispersiveNumCoeffs_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef COATING_DISPERSIVENUMCOEFFS_HIT_H
#define COATING_DISPERSIVENUMCOEFFS_HIT_H

#include "internal/optix_declarations.h"  // For RT_HOSTDEVICE
#include "randomGenerator.h"
#include "rayData.h"
#include "Coating_DispersiveNumCoeffs_hit.h"

/* declare class */
/**
  *\class   Coating_DispersiveNumCoeffs_ReducedParams 
  *\ingroup Coating
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
class Coating_DispersiveNumCoeffs_ReducedParams: public Coating_NumCoeffs_ReducedParams
{
public:

};

/**
 * \detail hitCoatingNumCoeff 
 *
 * modifies the raydata according to the parameters of the coating
 *
 * \param[in] rayStruct &ray, Mat_hitParams hitParams, const Coating_DispersiveNumCoeffs_ReducedParams& params
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitCoatingNumCoeff(rayStruct &ray, Mat_hitParams hitParams, const Coating_DispersiveNumCoeffs_ReducedParams& params)
{
	uint32_t x1[5];
	RandomInit(ray.currentSeed, x1); // init random variable
	double xi=Random(x1); // get uniform distribution between zero and one 
	ray.currentSeed=x1[4]; // save for next randomization
	bool coat_reflected=false;  // init the ray to be absorbed in the coating
	ray.running=false; // init ray to be absorbed
	double absorb=1-params.r+params.t; // probability of ray beeing absorbed
	// if xi is smaller than reflection coefficient, the ray is reflected
	if (xi<params.r)
	{
		// if reflection is not the most probable alternative, increase depth
		if ( (params.r<params.t) && (params.r<absorb) )
			ray.depth++;
		ray.running=true; // keep ray running
		coat_reflected=true; // indicate that ray was reflected in coating
	}
	else 
	{
		// if we have transmission
		if (xi<(params.r+params.t))
		{
			ray.running=true; // keep ray running
			if ( (params.t<params.r) && (params.t<absorb) )
				ray.depth++; // if transmission is not the most probable alternative, increase depth
		}
	}
	return coat_reflected;
};

#endif


