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

/**\file AsphericalSurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef ASPHERICALSURFACE_GEOMRENDER_INTERSECT_H
  #define ASPHERICALSURFACE_GEOMRENDER_INTERSECT_H
  
/* include header of basis class */
#include "Material_GeomRender_hit.h"
#include "../AsphericalSurface_intersect.h"
#include "../rayTracingMath.h"

/* declare class */
/**
  *\class   AsphericalSurface_GeomRender_ReducedParams 
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
class AsphericalSurface_GeomRender_ReducedParams : public AsphericalSurface_ReducedParams
{
  public:
};

/**
 * \detail calcHitParamsAsphere_GeomRender 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,AsphericalSurface_GeomRender_ReducedParams params
 * 
 * \return Mat_GeomRender_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_GeomRender_hitParams calcHitParamsAsphere_GeomRender(double3 position,AsphericalSurface_GeomRender_ReducedParams params)
{
	Mat_GeomRender_hitParams t_hitParams_GeomRender;
	Mat_hitParams t_hitParams;
	t_hitParams=calcHitParamsAsphere(position, params);
	t_hitParams_GeomRender.normal=t_hitParams.normal;
	return t_hitParams_GeomRender;

}

#endif
