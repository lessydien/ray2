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

/**\file PlaneSurface_GeomRender_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef PLANESURFACE_GEOMRENDER_INTERSECT_H
  #define PLANESURFACE_GEOMRENDER_INTERSECT_H

/* include header of basis class */
#include "Material_GeomRender_hit.h"
#include "../PlaneSurface_intersect.h"
#include "../rayTracingMath.h"

/* declare class */
/**
  *\class   PlaneSurface_GeomRender_ReducedParams 
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
class PlaneSurface_GeomRender_ReducedParams : public PlaneSurface_ReducedParams
{
  public:
};


#endif
