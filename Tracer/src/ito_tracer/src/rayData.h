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

/**\file raydata.h
* \brief collection of classes of the different types of rays for the various ray representations of the lightfield
* 
*           
* \author Mauch
*/

#ifndef RAYDATA_H
#define RAYDATA_H

#include "macrosim_types.h"
#include "MacroSimLib.h"

typedef enum 
{
  RAYPOS_RAND_RECT,
  RAYPOS_RAND_RECT_NORM,
  RAYPOS_GRID_RECT,
  RAYPOS_RAND_RAD,
  RAYPOS_RAND_RAD_NORM,
  RAYPOS_GRID_RAD,
  RAYPOS_UNKNOWN
} rayPosDistrType;

typedef enum 
{
  RAYDIR_RAND_RECT,
  RAYDIR_RAND_RAD,
  RAYDIR_RANDNORM_RECT,
  RAYDIR_RANDIMPAREA,
  RAYDIR_UNIFORM,
  RAYDIR_GRID_RECT,
  RAYDIR_GRID_RECT_FARFIELD,
  RAYDIR_GRID_RAD,
  RAYDIR_UNKNOWN
} rayDirDistrType;

typedef enum
{
	ACCEL_NOACCEL,
	ACCEL_BVH,
	ACCEL_SBVH,
	ACCEL_MBVH,
	ACCEL_LBVH,
	ACCEL_TRIANGLEKDTREE
} accelType;


/* declare class */
/**
  *\class   rayStructBase
  *\brief   base class of the classes that define the datasrtuctures of all the raytypes defined in this application
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
class rayStructBase
{
public:
  double3 position;
  double3 direction;
  double  opl;  //!> optical path length
  double  lambda;
  double  nImmersed; //!> refractive index of material the ray is immersed in
  double   flux;
  int     depth; //!> gives the number of times the ray did depart from the most probable way, e.g. was partially reflected at a mainly transmitting surface or scattered
  int     currentGeometryID;
  bool	running;
  unsigned int currentSeed;
};

/* declare class */
/**
  *\class   rayStruct
  *\brief   class that defines the datastructure of the geometric ray
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
class rayStruct  : public rayStructBase
{
public:
}; 

/* declare class */
/**
  *\class   diffRayStruct
  *\brief   classes that defines the datastructure of the differential ray
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
class diffRayStruct : public rayStruct
{
public:
  double3 mainDirX;
  double3 mainDirY;
  double2 wavefrontRad;
};

/* declare class */
/**
  *\class   rayStruct_PathTracing
  *\brief   classes that defines the datastructure of the ray for path tracing
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
class rayStruct_PathTracing : public rayStruct
{
public:
  double result;
  short secondary_nr;
  bool secondary;
};

/* declare class */
/**
  *\class   gaussBeamRayStruct
  *\brief   classe that defines the datastructures of the ray representation of a gaussian beam
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
class gaussBeamRayStruct : public rayStructBase
{
public:
  rayStruct baseRay, waistRayX, waistRayY, divRayX, divRayY;
};

typedef struct
{
  double t_baseRay, t_waistRayX, t_waistRayY, t_divRayX, t_divRayY;
} gaussBeam_t;

typedef struct
{
  double3 normal_baseRay, normal_waistRayX, normal_waistRayY, normal_divRayX, normal_divRayY;
} gaussBeam_geometricNormal;

//typedef enum 
//{
//  SIM_GEOMRAYS_NONSEQ,
//  TRACE_SEQ,
//  SIM_GAUSSBEAMRAYS_NONSEQ,
//  TRACE_SEQ,
//  SIM_DIFFRAYS_NONSEQ,
//  TRACE_SEQ
//} traceMode;

#endif
