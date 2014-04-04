/**\file MaterialParams.h
* \brief header file that contains the material params
* 
*           
* \author Mauch
*/

#ifndef MATERIALPARAMS_H
  #define MATERIALPARAMS_H

#include "Scatter_hit.h"
#include "Coating_hit.h"
#include "Geometry_Intersect.h"

#define PARSERR_LENGTH 64
#define NR_DOE_COEFFS 44

typedef enum 
{
  MT_MIRROR,
  MT_NBK7,
  MT_BK7,
  MT_AIR,
  MT_ABSORB,
  MT_COVGLASS,
//  MT_TORSPARR1D,
  MT_LINGRAT1D,
  MT_REFRMATERIAL,
  MT_DOE,
  MT_IDEALLENSE,
  MT_DIFFRACT,
  MT_FILTER,
  MT_PATHTRACESRC,
  MT_VOLUMESCATTER,
  MT_VOLUMESCATTERBOX,
  MT_VOLUMEABSORB,
  MT_RENDERLIGHT,
  MT_UNKNOWNMATERIAL
} MaterialType;

typedef enum
{
  MAT_DISPFORMULA_SCHOTT,
  MAT_DISPFORMULA_SELLMEIER1,
  MAT_DISPFORMULA_UNKNOWN,
  MAT_DISPFORMULA_NODISP
} MaterialDispersionFormula;

typedef struct
{
	int dispersionFormulaIndex;
	double lambdaMin;
	double lambdaMax;
	double paramsNom[6];
	double paramsDenom[6];
	char errMsg[PARSERR_LENGTH];
} parseGlassResultStruct;

class parseDOEResultStruct
{
public:
	~parseDOEResultStruct()
	{
		free(errMsg);
		free(coeffArray);
	}
	double coeffArray[NR_DOE_COEFFS];
	unsigned short coeffLength;
	char errMsg[PARSERR_LENGTH];
};

class ParseGratingResultStruct
{
  public:
	  /* destructor */
    ~ParseGratingResultStruct()
    {
		free(lambdaPtr);
		free(diffOrdersPtr);
		free(RTP01Ptr);
		free(RTP10Ptr);
		free(RTS01Ptr);
		free(RTS10Ptr);
		free(errMsg);
	}
	int nrWavelengths;
	double *lambdaPtr;
	int nrOrders;
	short *diffOrdersPtr;
	double *RTP01Ptr;
	double *RTP10Ptr;
	double *RTS01Ptr;
	double *RTS10Ptr;
	double g;
	char errMsg[PARSERR_LENGTH];
};

typedef struct
{
	MaterialType matType;
	ScatterType scatterType;
	CoatingType coatingType;
	double2 nRefr;
	char glassName[GEOM_CMT_LENGTH];
	char immersionName[GEOM_CMT_LENGTH];
	char fileName_diffractionLookUp[GEOM_CMT_LENGTH];
	short nrDiffOrders;
	short diffOrder[MAX_NR_DIFFORDERS];
	double diffEff[MAX_NR_DIFFORDERS];
	double gratingConstant;
	bool gratingOrdersFromFile;
	bool gratingLinesFromFile;
	bool gratingEffsFromFile;
	double varParams[MAX_NR_MATPARAMS];
	double3 scatteringAxis;
	double3 coatingAxis;
    double coating_r;
    double coating_t;
	double coating_a_r;
	double coating_c_r;
	double coating_a_t;
	double coating_c_t;
	double idealLense_f0;
	double idealLense_lambda0;
	double idealLense_A;
	bool importanceArea;
	double3 importanceAreaTilt;
	double3 importanceAreaRoot;
	double2 importanceAreaHalfWidth;
	ApertureType importanceAreaApertureType;
	int importanceObjNr;
	double2 importanceConeAlphaMin;
	double2 importanceConeAlphaMax;
	double3 root;
	double3 normal;
	double2 apertureHalfWidth;
	int covGlassGeomID;
	double filterMax;
	double filterMin;
	double flux; //!> flux for source material in path tracing
	double2 srcAreaHalfWidth;
	double3 srcAreaRoot;
	double3 srcAreaTilt;
	ApertureType srcAreaType;
	double3 tilt;
} MaterialParseParamStruct;

/* declare class */
/**
  *\class   MatDispersionParams 
  *\brief   full set of params that is describing the chromatic behaviour of the material properties
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
class MatDispersionParams
{
public:

};


#endif
