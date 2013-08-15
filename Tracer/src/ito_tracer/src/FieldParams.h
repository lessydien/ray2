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

/**\file FieldParams.h
* \brief header file that contains the fieldParams class
* 
*           
* \author Mauch
*/

#ifndef FIELDPARAMS_H
  #define FIELDPARAMS_H

#include "rayData.h"
#include "MacroSimLib.h"

typedef enum 
{
  FIELD_NO_ERR,
  FIELD_ERR,
  FIELD_INDEXOUTOFRANGE_ERR
} fieldError;

//typedef enum
//{
//	metric_m,
//	metric_mm,
//	metric_mu,
//	metric_nm,
//	metric_au // arbitray units
//} metric_unit;
//
//typedef struct
//{
//	metric_unit x;
//	metric_unit y;
//	metric_unit z;
//} axesUnits;
//
//typedef enum 
//{
//  GEOMRAYFIELD,
//  DIFFRAYFIELD,
//  DIFFRAYFIELDRAYAIM,
//  PATHTRACERAYFIELD,
//  SCALARFIELD,
//  VECFIELD,
//  INTFIELD,
//  FIELDUNKNOWN
//} fieldType;

typedef struct FieldParseParamStruct
{
  geometry_type type;
  unsigned long long width; //!> in grid rect: number of rays along x-axis; in grid rad: number of rays along radial direction
  unsigned long long height; //!> in grid rect: number of rays along y-axis; in grid rad: number of rays along angular direction
  unsigned long long widthLayout; //!> analog to width for LayoutMode
  unsigned long long heightLayout; //!> analog to height for LayoutMode
  double power;
  rayDirDistrType rayDirDistr;
  rayPosDistrType rayPosDistr;
  double3 rayDirection;
  double lambda;
  double2 apertureHalfWidth1;
  double2 apertureHalfWidth2;
//  double rotNormal1;
  MaterialParseParamStruct materialParams;
  double3 root;
  double3 normal;
  double2 alphaMax; //!> maximum angle relative to source normal of the cone into which raydirections are uniformly distributed along x and y
  double2 alphaMin; //!> minimum angle relative to source normal of the cone into which raydirections are uniformly distributed along x and y
  double coherence;
  ulong2 nrRayDirections; //!> nr of rays that are shot in random directions from each point source in case of diiferential ray source
  double epsilon; //!> short distance that the rays are moved from the caustic of each point source in case of differential ray sources
  double3 tilt;
  int importanceObjNr;
  double2 importanceConeAlphaMax;
  double2 importanceConeAlphaMin;
  double3 importanceAreaTilt; 
  double3 importanceAreaRoot;
  double2 importanceAreaHalfWidth;
  ApertureType importanceAreaApertureType;
  bool importanceArea;
//	int importanceObjNr;

} ;

/* declare class */
/**
  *\class   fieldParams
  *\ingroup Field
  *\brief   
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
class fieldParams
{
public:
	long3 nrPixels; // number of pixels in each direction
	double4x4 MTransform; 
	double3 scale; // physical size of voxel in each dimension
	axesUnits units;
	double lambda;
	metric_unit unitLambda;

	/* standard constructor */
	//fieldParams()
	//{
	//	this->nrPixels=make_long3(0,0,0);
	//	this->MTransform=make_double4x4(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);
	//	this->scale=make_double3(0,0,0);
	//	this->unitLambda=metric_au;
	//	axesUnits l_axesUnits;
	//	l_axesUnits.x=metric_au;
	//	l_axesUnits.y=metric_au;
	//	l_axesUnits.z=metric_au;
	//	this->units=l_axesUnits;
	//}
	/* destructor */
	//~fieldParams()
	//{
	//}
};

#endif
