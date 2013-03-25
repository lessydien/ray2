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

/**\file MaterialLinearGrating1D.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALLINEARGRATING1D_H
#define MATERIALLINEARGRATING1D_H

#include "Material.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "Group.h"
#include "MaterialLinearGrating1D_hit.h"
#include "MaterialRefracting.h"
#include <stdlib.h>

#define PATH_TO_HIT_LINEARGRATING1D "ITO-MacroSim_generated_hitFunctionLinGrat1D"

/* declare class */
/**
  *\class   MaterialLinearGrating1D_DiffractionParams
  *\ingroup Material
  *\brief   full set of params describing the chromatic behaviour of the diffraction properties.
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
class MatLinearGrating1D_DiffractionParams
{
public:
	MatLinearGrating1D_DiffractionParams()
	{
		nrWavelengths=0;
		lambdaPtr=NULL;
		nrOrders=0;
		nrOrdersSim=0;
		diffOrdersPtr=NULL;
		RTP01Ptr=NULL;
		RTP10Ptr=NULL;
		RTS01Ptr=NULL;
		RTS10Ptr=NULL;
		g=0;
		diffAxis=make_double3(0,0,0);
	}
	  /* destructor */
    ~MatLinearGrating1D_DiffractionParams()
    {
		if (diffOrdersPtr != NULL)
			delete diffOrdersPtr;
		if (RTP01Ptr != NULL)
			delete RTP01Ptr;
		if (RTP10Ptr != NULL)
			delete RTP10Ptr;
		if (RTS01Ptr != NULL)
			delete RTS01Ptr;
		if (RTS10Ptr != NULL)
			delete RTS10Ptr;
	}
	int nrWavelengths;
	double *lambdaPtr;
	short nrOrders; // number of orders of which efficiency data is present
	short nrOrdersSim; // number of orders that are to be accounted for in the current simulation
	short *diffOrdersPtr;
	double *RTP01Ptr;
	double *RTP10Ptr;
	double *RTS01Ptr;
	double *RTS10Ptr;
	double g;
	double3 diffAxis;
};

//class MatRefracting_DispersionParams
//{
//public:
//	MaterialDispersionFormula dispersionFormula;
//	double lambdaMin;
//	double lambdaMax;
//	double paramsNom[5];
//	double paramsDenom[5];
//};

/* declare class */
/**
  *\class   MaterialLinearGrating1D
  *\ingroup Material
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
class MaterialLinearGrating1D: public Material
{
	protected:
		MatLinearGrating1D_params params;
		MatLinearGrating1D_DiffractionParams* diffractionParamsPtr;
		MatRefracting_DispersionParams* glassDispersionParamsPtr;
		MatRefracting_DispersionParams* immersionDispersionParamsPtr;
		char glassName[GEOM_CMT_LENGTH];
		MaterialError calcRefrIndices(double lambda);
		MaterialError calcDiffEffs(double lambda);
		double lambda_old;

  public:
    /* standard constructor */
    MaterialLinearGrating1D()
	{
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, PATH_TO_HIT_LINEARGRATING1D );
//		this->setPathToPtx(path_to_ptx);
	}
    MaterialLinearGrating1D(int nrWave, int nrOrd )
	{
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, PATH_TO_HIT_LINEARGRATING1D );
//		this->setPathToPtx(path_to_ptx);

	}
	/* destrcuctor */
	~MaterialLinearGrating1D()
	{
		if (diffractionParamsPtr != NULL)
		{
			delete this->diffractionParamsPtr;
			this->diffractionParamsPtr=NULL;
		}
		if (this->glassDispersionParamsPtr!=NULL)
			delete glassDispersionParamsPtr;
//		delete path_to_ptx;
	}
	void setParams(MatLinearGrating1D_params paramsIn);
	MatLinearGrating1D_params getParams(void);

    MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	MaterialError createCPUSimInstance(double lambda);
	MaterialError updateCPUSimInstance(double lambda);
//	void setPathToPtx(char* path);
//	char* getPathToPtx(void);

	void hit(rayStruct &ray, Mat_hitParams hitParams, double t_hit, int geometryID);
	void hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal, int geometryID);
	MatRefracting_DispersionParams* getGlassDispersionParams(void);
	void setGlassDispersionParams(MatRefracting_DispersionParams* dispersionParamsInPtr);
	MatRefracting_DispersionParams* getImmersionDispersionParams(void);
	void setImmersionDispersionParams(MatRefracting_DispersionParams* dispersionParamsInPtr);
	MatLinearGrating1D_DiffractionParams* getDiffractionParams(void);
	void setDiffractionParams(MatLinearGrating1D_DiffractionParams* diffractionParamsInPtr);
	MaterialError processParseResults(MaterialParseParamStruct &parseResults_MatPtr, parseGlassResultStruct* parseResults_GlassPtr, parseGlassResultStruct* parseResults_ImmPtr, ParseGratingResultStruct* parseResults_GratPtr);
	MaterialError parseXml(pugi::xml_node &geometry);	

};

#endif


