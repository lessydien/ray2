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

/**\file SphericalSurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef SPHERICALSURFACE_GEOMRENDER_INTERSECT_H
  #define SPHERICALSURFACE_GEOMRENDER_INTERSECT_H
  
/* include geometry lib */
#include "Material_GeomRender_hit.h"
#include "../SphericalSurface_intersect.h"
#include "../rayTracingMath.h"

/* declare class */
/**
  *\class   SphericalSurface_GeomRender_ReducedParams
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
class SphericalSurface_GeomRender_ReducedParams : public SphericalSurface_ReducedParams
{
	public:
};

#endif
