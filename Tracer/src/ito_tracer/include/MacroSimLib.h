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

#ifndef MACROSIMTRACER_H
	#define MACROSIMTRACER_H

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "PropagationMath.h"

using namespace std;

typedef enum
{
	metric_m,
	metric_mm,
	metric_mu,
	metric_nm,
	metric_au // arbitray units
} metric_unit;

typedef struct
{
	metric_unit x;
	metric_unit y;
	metric_unit z;
} axesUnits;

typedef enum 
{
  GEOMRAYFIELD,
  DIFFRAYFIELD,
  DIFFRAYFIELDRAYAIM,
  PATHTRACERAYFIELD,
  SCALARFIELD,
  SCALARUSERWAVE,
  SCALARSPHERICALWAVE,
  SCALARPLANEWAVE,
  SCALARGAUSSIANWAVE,
  VECFIELD,
  INTFIELD,
  PATHINTTISSUERAYFIELD,
  FIELDUNKNOWN
} fieldType;

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
class ItomFieldParams
{
public:
	long nrPixels[3]; // number of pixels in each direction
	double MTransform[16]; 
	double scale[3]; // physical size of voxel in each dimension
	axesUnits units;
	double lambda;
	metric_unit unitLambda;
	fieldType type;
};

typedef enum 
{
  SIM_GEOMRAYS_NONSEQ,
  SIM_GEOMRAYS_SEQ,
  SIM_GAUSSBEAMRAYS_NONSEQ,
  SIM_GAUSSBEAMRAYS_SEQ,
  SIM_DIFFRAYS_NONSEQ,
  SIM_DIFFRAYS_SEQ
} TraceMode;

typedef struct
{
	TraceMode traceMode;
} SimParams;

/* declare class */
/**
  *\class   simAssParams 
  *\ingroup SimAssistant
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
class simAssParams
{
public:
  bool RunOnCPU;
  SimParams simParams;
};

class RayPlotDataParams
{
public:
	RayPlotDataParams()
	{
	}
	~RayPlotDataParams()
	{
	}
	long subsetDim[2];
	unsigned long long launchOffset[2];
	unsigned long long launchDim[2];
};

class MacroSimTracerParams
{
public:
	MacroSimTracerParams()
	{
	}
	~MacroSimTracerParams()
	{
	}
	unsigned long long subsetWidth;
	unsigned long long subsetHeight;
	int numCPU;
	char glassFilePath[512];
	char inputFilesPath[512];
	char outputFilesPath[512];
	char path_to_ptx[512];
	TraceMode mode;
};

class MacroSimTracer
{
public:
	MacroSimTracer()
	{
	}
	~MacroSimTracer()
	{
	}

	bool runMacroSimRayTrace(char *xmlInput, void** fieldOut_ptrptr, ItomFieldParams* fieldOutParams, void* p2CallbackObject, void (*callbackProgress)(void* p2Object, int progressValue));
	bool runMacroSimLayoutTrace(char *xmlInput, void* p2CallbackObject, void (*callbackRayPlotData)(void* p2Object, double* rayPlotData, RayPlotDataParams *params));
	bool runConfPointSensorSim(ConfPoint_Params &params, double** res_ptrptr);
	bool checkVisibility(char *objectInput_filename);
	ostream* getCout();
};

#endif