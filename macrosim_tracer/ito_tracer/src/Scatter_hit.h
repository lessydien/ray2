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

#ifndef SCATTER_HIT_H
#define SCATTER_HIT_H

#include "Material_hit.h"
#include "Geometry_Intersect.h"

typedef enum 
{
  ST_NOSCATTER,
  ST_TORRSPARR1D,
  ST_TORRSPARR2D,
  ST_TORRSPARR2DPATHTRACE,
  ST_DOUBLECAUCHY1D,
  ST_DISPDOUBLECAUCHY1D,
  ST_LAMBERT2D,
  ST_PHONG,
  ST_COOKTORRANCE,
  ST_UNKNOWNSCATTER
} ScatterType;

//#include "internal/optix_declarations.h"  // For RT_HOSTDEVICE

/* declare class */
/**
  *\class   Scatter_ReducedParams
  *\ingroup Scatter
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
class Scatter_ReducedParams
{
public:
	double2 impAreaHalfWidth;
	double3 impAreaRoot;
	double3 impAreaTilt;
	ApertureType impAreaType;
};

//inline RT_HOSTDEVICE double hitScatter(rayStruct &ray)
//{
//	ray.running=false; // stop the reay
//	return 1;
//}

#endif


