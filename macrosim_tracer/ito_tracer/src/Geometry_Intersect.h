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

/**\file Geometry_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef GEOMETRYINTERSECT_H
  #define GEOMETRYINTERSECT_H

#include "Material_hit.h"


typedef enum 
{
  AT_RECT,
  AT_ELLIPT,
  AT_RECTOBSC,
  AT_ELLIPTOBSC,
  AT_INFTY,
  AT_UNKNOWNATYPE
} ApertureType;

typedef enum
{
	IMP_FARFIELD,
	IMP_CONV,
	IMP_UNKNOWN
} ImpAreaType;

typedef enum 
{
  SRC,
  DIFFSRC,
  DIFFSRC_RAYAIM,
  DIFFSRC_FREEFORM,
  DIFFSRC_HOLO,
  PATHTRACESRC,
  GEOM_PLANESURF,
  GEOM_SPHERICALSURF,
  GEOM_PARABOLICSURF,
  GEOM_SPHERICALLENSE,
  GEOM_MICROLENSARRAYSURF,
  GEOM_APERTUREARRAYSURF,
  GEOM_STOPARRAYSURF,
  GEOM_MICROLENSARRAY,
  GEOM_ASPHERICALSURF,
  GEOM_CYLLENSESURF,
  GEOM_GRATING,
  GEOM_CYLPIPE,
  GEOM_CYLLENSE,
  GEOM_CONEPIPE,
  GEOM_IDEALLENSE,
  GEOM_APERTURESTOP,
  GEOM_COSINENORMAL,
  GEOM_CADOBJECT,
  GEOM_SUBSTRATE,
  GEOM_VOLUMESCATTERERBOX,
  GEOM_UNKNOWN
} geometry_type;

/* declare class */
/**
  *\class   Geometry_ReducedParams
  *\ingroup Geometry
  *\brief   base class of the reduced params of all geometries
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
class Geometry_ReducedParams
{
public:
	int geometryID;
	double3 tilt;
};

#endif