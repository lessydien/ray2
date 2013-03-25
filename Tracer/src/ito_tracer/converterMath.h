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

#ifndef CONVERTERMATH_H
#define CONVERTERMATH_H

//#include "internal/optix_declarations.h"  // For RT_HOSTDEVICE
#include "my_vector_types.h"
#include <optix_math.h>
#include "GlobalConstants.h"
#include "rayData.h"
//#include "complex.h"
#include <complex>
#include "Converter.h"
#include "math.h"

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

int doGaussElimOn3x3(double3x3 *Matrix);

/* direct translation from matlab implementation calcFieldFromGeometricBeamlets.m */
bool gaussBeams2ScalarFieldCPU(gaussBeamRayStruct *rayList, unsigned long long rayListLength, complex<double>* U, fieldParams* params);
bool geomRays2IntensityCPU(rayStruct* rayListPtr, unsigned long long rayListLength, double* IntensityPtr, double4x4 MTransform, double3 scale, long3 nrPixels, double coherence);

#endif