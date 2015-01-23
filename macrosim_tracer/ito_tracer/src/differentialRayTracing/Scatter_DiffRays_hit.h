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

#ifndef SCATTER_DIFFRAYS_HIT_H
#define SCATTER_DIFFRAYS_HIT_H

#include "Material_DiffRays_hit.h"
#include "../Scatter_hit.h"

//#include "internal/optix_declarations.h"  // For RT_HOSTDEVICE

/* declare class */
/**
  *\class   Scatter_ReducedParams
  *\brief   base class of reduced params of scatters defined in this application
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
class Scatter_DiffRays_ReducedParams : public Scatter_ReducedParams
{
public:

};

//inline RT_HOSTDEVICE double hitScatter(rayStruct &ray)
//{
//	ray.running=false; // stop the reay
//	return 1;
//}

#endif


