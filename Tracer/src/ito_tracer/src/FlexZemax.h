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

/**\file FlexZemax.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef FLEXZEMAX_H
#define FLEXZEMAX_H

#include <map>
#include <string>

#include "macrosim_types.h"
#include "macrosim_functions.h"
//#include "rayTracingMath.h"
#include "Material.h"
#include "MaterialLib.h"
#include "differentialRayTracing/MaterialLib_DiffRays.h"
#include "differentialRayTracing/Material_DiffRays.h"
#include "DetectorLib.h"
#include "FieldLib.h"
#include "PupilLib.h"
#include "CoatingLib.h"
#include "differentialRayTracing/CoatingLib_DiffRays.h"
#include "differentialRayTracing/Coating_DiffRays.h"
#include "ScatterLib.h"
#include "differentialRayTracing/ScatterLib_DiffRays.h"
#include "differentialRayTracing/Scatter_DiffRays.h"
#include <string.h>
#include "Geometry_intersect.h"
#include "Geometry.h"
#include "GeometryLib.h"
#include "differentialRayTracing/GeometryLib_DiffRays.h"
#include "GlobalConstants.h"
#include "rayData.h"
#include "SimAssistantLib.h"

using namespace std;

#ifndef PI
	#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

typedef enum 
{
  // sequential surface types
  OT_STANDARD,
  OT_CRDBRK,
  OT_DGRATING,
  OT_EVENASPH,
  OT_ODDASPH,
  OT_ASPHSURF,
  // nonsequential object types
  OT_SRC,
  OT_BICONLENSE,
  OT_DIFFSRC,
  OT_DIFFSRC_RAYAIM,
  OT_DIFFSRC_FREEFORM,
  OT_DIFFSRC_HOLO,
  OT_PATHTRACINGSRC,
  OT_IDEALLENSE,
  OT_DET,
  OT_STLENSE,
  OT_CYLPIPE,
  OT_CONEPIPE,
  OT_APERTURESTOP,
  OT_COSINENORMAL,
  OT_ILLPUPIL,
  OT_UNKNOWNOTYPE
} ObjectType;

typedef enum
{
	SIMTYPE_GEOM_RAYS,
	SIMTYPE_DIFF_RAYS,
	SIMTYPE_PATHTRACING,
	SIMTYPE_UNKNOWN
} SimulationType;


typedef enum 
{
  TILTDEC,
  DECTILT
} OrderType;

typedef enum 
{
  BEFORESURF,
  AFTERSURF
} DecenterType;

typedef enum 
{
  PARSER_ERR,
  PARSER_NO_ERR
} parserError;

//union variant_u
//{
//	double d;
//	double2 d2;
//	double3 d3;
//	float f;
//	float2 f2;
//	float3 f3;
//	unsigned int ui;
//	uint2 ui2;
//	uint3 ui3;
//	signed int si;
//	int2 si2;
//	int3 si3;
//	unsigned long ul;
//	ulong2 ul2;
//	ulong3 ul3;
//	signed long sl;
//	unsigned long long ull;
//	signed long long sll;
//	short s;
//	bool b;
//	char c;
//	SimulationType simType;
//	ObjectType objType;
//	OrderType order;
//	ApertureType aperture;
//	ScatterType scatter;
//	CoatingType coating;
//	MaterialType material;
//};

enum type_enum
{
	TYPE_CHAR,
	TYPE_UINT,
	TYPE_UINT2,
	TYPE_UINT3,
	TYPE_FLOAT,
	TYPE_FLOAT2,
	TYPE_FLOAT3,
	TYPE_DOUBLE,
	TYPE_DOUBLE2,
	TYPE_DOUBLE3,
	TYPE_SINT,
	TYPE_SINT2,
	TYPE_SINT3,
	TYPE_ULONG,
	TYPE_ULONG2,
	TYPE_ULONG3,
	TYPE_SLONG,
	TYPE_SLONG2,
	TYPE_SLONG3,
	TYPE_SLONGLONG,
	TYPE_ULONGLONG,
	TYPE_SHORT,
	TYPE_BOOL,
	TYPE_SIMTYPE,
	TYPE_OBJTYPE,
	TYPE_APERTRUETYPE,
	TYPE_SCATTERTYPE,
	TYPE_COATINGTYPE,
	TYPE_MATERIALTYPE,
};

//struct variant
//{
//	variant_u value;
//	type_enum type;
//};


typedef struct 
{
  double2 nRefr; // refractive indices
  double3 root;
  double3 dec1;
  double3 tilt1;
  double3 dec2;
  double3 tilt2;
  OrderType order1;
  OrderType order2;
  ApertureType aperture;
  double2 apertureHalfWidth1;
  double2 apertureHalfWidth2;
  double2 obscurationHalfWidth;
  // material params
  double lines;
  short nrDiffOrders;
  short diffOrder[MAX_NR_DIFFORDERS];
  double diffEff[MAX_NR_DIFFORDERS];
  double params[8]; // used for example to store the asphere coefficients of an aspher
  double asphereParams[MAX_NR_ASPHPAR];
  double matParams[MAX_NR_MATPARAMS]; // used for example to store parameters of scattering surfaces
  bool gratingOrdersFromFile;
  bool gratingLinesFromFile;
  bool gratingEffsFromFile;
  // coating params
  double coating_side_r;
  double coating_side_t;
  double coating_side_a_r;
  double coating_side_c_r;
  double coating_side_a_t;
  double coating_side_c_t;
  double coating_front_r;
  double coating_front_t;
  double coating_front_a_r;
  double coating_front_c_r;
  double coating_front_a_t;
  double coating_front_c_t;
  double coating_back_r;
  double coating_back_t;
  double coating_back_a_r;
  double coating_back_c_r;
  double coating_back_a_t;
  double coating_back_c_t;
  // scatter params
  ScatterType scatterFront; // scatter of front surface
  CoatingType coatingFront; // coating of front surface
  ScatterType scatterBack; // ...
  CoatingType coatingBack;
  ScatterType scatterSide;
  CoatingType coatingSide;
  // source params
  char immersionName[GEOM_CMT_LENGTH];
  long long rayFieldHeight;
  long long rayFieldWidth;
  long long rayFieldHeightLayout;
  long long rayFieldWidthLayout;
  double power;
  rayDirDistrType rayDirDistr;
  rayPosDistrType rayPosDistr;
  double2 rayDirectionTilt;
  double2 alphaMax;
  double2 alphaMin;
  double lambda;
  // detector params
  bool detector;
  ulong2 detPixel;
  ulong2 detPixel_PhaseSpace;
  double2 dirHalfWidth;
  detType detectorType;
  double idealLense_f0;
  double idealLense_lambda0;
  double idealLense_A;
  double srcCoherence;
  ulong2 nrRayDirections;
  double epsilon;
  int importanceObjNr;
  double2 importanceConeAlphaMax;
  double2 importanceConeAlphaMin;
  bool importanceArea;
  double3 cosNormAxis;
  double cosNormAmpl;
  double cosNormPeriod;
  int covGlassGeomID;
  double filterMax;
  double filterMin;
  bool sweep;
  ulong2 nrRaysPerPixel; //!> number of rays that are shot from each detector pixel in reverse path tracing
  double iterationAccuracy;
} ZemaxParamDetailStruct;

typedef struct 
{
  ObjectType type;
  char comment[GEOM_CMT_LENGTH];
  char glassName[GEOM_CMT_LENGTH];
  char fileNameDiffraction[GEOM_CMT_LENGTH];
  double2  radius1;
  double2  radius2;
  double  thickness;
  MaterialType glass;
  double  diameter;
  double  conic1;
  double  conic2;
  ZemaxParamDetailStruct details;
} ZemaxParamStruct;

typedef struct
{
  GeometryParseParamStruct* geometryParams;
  FieldParseParamStruct* sourceParams;
  DetectorParseParamStruct* detectorParams;
  PupilParseParamStruct* pupilParams;
  long long detectorNumber;
  long long sourceNumber;
  long long geomNumber;
  long long pupilNumber;
  char errMsg[PARSERR_LENGTH];
  SimulationType simType;
} parseResultStruct;


parserError parseDOEFile(parseDOEResultStruct** parseResultsDOEPtrPtr, FILE *hfile, int DOEnr);
parserError parseMicroSimGratingData(ParseGratingResultStruct** parseResultsGratingPtrPtr, FILE *hfile);
parserError parseZemaxGlassCatalogOld(parseGlassResultStruct** parseResultsGlassPtrPtr, FILE *hfile, char *glassName);
parserError parseZemaxGlassCatalog(parseGlassResultStruct** parseResultsGlassPtrPtr, FILE *hfile, const char *glassName);
parserError parseZemaxPrescr(parseResultStruct** parseResultsTest,  FILE *hfile, SimMode mode);
parserError initSurfaceStruct(ZemaxParamStruct* surface);
void initGeomParseFlags(void);
bool checkObjectDefinition(ZemaxParamStruct *objectPtr);

#endif