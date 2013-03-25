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

/**\file DetectorParams.h
* \brief header file that contains the detectorParams class
* 
*           
* \author Mauch
*/

#ifndef DETECTORPARAMS_H
  #define DETECTORPARAMS_H

#include "Geometry_Intersect.h"
#include "rayData.h"

typedef enum
{
	DET_OUT_MAT,
	DET_OUT_TEXT,
	DET_OUT_X3P
} detOutFormat;

typedef enum 
{
  DET_RAYDATA,
  DET_RAYDATA_RED,
  DET_RAYDATA_GLOBAL,
  DET_RAYDATA_RED_GLOBAL,
  DET_INTENSITY,
  DET_PHASESPACE,
  DET_FIELD,
  DET_UNKNOWN
} detType;

typedef struct
{
  ulong2 detPixel;
  ulong2 detPixel_PhaseSpace;
  detType detectorType;
  double2 apertureHalfWidth;
//  double rotNormal;
  double3 root;
  double3 normal;
  long long geomID;
  double3 tilt;
  int importanceObjNr;
  double2 importanceConeAlphaMax;
  double2 importanceConeAlphaMin;
  double3 importanceAreaTilt; 
  double3 importanceAreaRoot;
  double2 importanceAreaHalfWidth;
  ApertureType importanceAreaApertureType;
  bool importanceArea;
  ulong2 nrRaysPerPixel;
  rayDirDistrType rayDirDistr;
  rayPosDistrType rayPosDistr;
  double2 dirHalfWidth;
} DetectorParseParamStruct;

/* declare class */
/**
  *\class   detParams 
  *\ingroup Detector
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
class detParams
{
public:

  ulong2 detPixel;
  double2 apertureHalfWidth;
//  double rotNormal;
  double3 tilt;
  double3 root;
  double3 normal;
  double4x4 MTransform;
  long long geomID;
  detOutFormat outFormat;
  int subSetNr;
  char *filenamePtr;
};


#endif
