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

/**\file Scatter_CookTorrance_hit.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef SCATTER_COOKTORRANCE_GEOMRENDER_HIT_H
  #define SCATTER_COOKTORRANCE_GEOMRENDER_HIT_H
  
#include "../randomGenerator.h"
#include "../rayTracingMath.h"
#include "../rayData.h"
#include "../Scatter_CookTorrance_hit.h"
#include "Material_GeomRender_hit.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif


/**
 * \detail hitCookTorrance_GeomRender
 *
 * modifies the raydata according to the parameters of the scatter
 *
 * \param[in] rayStruct &prd, Mat_hitParams hitParams, ScatCookTorrance_GeomRender_params params
 * 
 * \return bool
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE bool hitCookTorrance_GeomRender(geomRenderRayStruct &prd, Mat_hitParams hitParams, ScatCookTorrance_params params)
{
	// do the geometric hit
	if (!hitCookTorrance(prd, hitParams, params) )
		return false;

	return true;
};

#endif
