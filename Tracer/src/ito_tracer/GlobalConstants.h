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

/**\file GlobalConstants.h
* \brief defines of values that need to be known everywhere in the application
* 
*           
* \author Mauch
*/

// file paths
#ifdef MACROSIMSTART
	#define EXTERN
#else
	#define EXTERN extern
#endif
EXTERN char FILE_GLASSCATALOG[512];
EXTERN char OUTPUT_FILEPATH[512];
EXTERN char INPUT_FILEPATH[512];

#ifndef GLOBALCONSTANTS_H
  #define GLOBALCONSTANS_H extern

//constants for raytracing on CPU
extern int MAX_DEPTH_CPU;
//#define MAX_DEPTH_CPU 10
extern const float MIN_FLUX_CPU;
extern const double MAX_TOLERANCE;

//#define INPUT_FILEPATH "C:\\mauch\\MacroSim_In"
#define PATH_SEPARATOR "\\"

#define GEOM_CMT_LENGTH 64 // length of comment of geometries
#define MAX_NR_DIFFORDERS 9 // number of diffraction orders for diffraction gratings
#define MAX_NR_MATPARAMS 15
#define MAX_NR_ASPHPAR 15

#define DOUBLE_MAX 9223372036854775807
#define EPSILON 0.00000001
//#define DIFF_EPSILON 0.005

#define PI ((double)3.141592653589793238462643383279502884197169399375105820)

//#define MAX_GPURAYCOUNT 4000000
//#define GPU_SUBSET_WIDTH_MAX 2000
//#define GPU_SUBSET_HEIGHT_MAX 2000

//#define MAX_DEPTH_GPU 10

#endif
