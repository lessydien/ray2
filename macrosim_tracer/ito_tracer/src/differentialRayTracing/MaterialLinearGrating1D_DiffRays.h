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

/**\file MaterialLinearGrating1D_DiffRays.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef MATERIALLINEARGRATING1D_DIFFRAYS_H
#define MATERIALLINEARGRATING1D_DIFFRAYS_H

#include "Material_DiffRays.h"
#include <sampleConfig.h>
#include <stdio.h>
#include "../Group.h"
#include "../MaterialLinearGrating1D.h"
#include "MaterialLinearGrating1D_DiffRays_hit.h"
#include "MaterialRefracting_DiffRays.h"
#include <stdlib.h>

#define PATH_TO_HIT_LINEARGRATING1D_DIFFRAYS "macrosim_tracer_generated_hitFunctionLinGrat1D"

///* declare class */
///**
//  *\class   MaterialLinearGrating1D_DiffRays_DiffractionParams
//  *\brief   full set of params describing the chromatic behaviour of the diffraction properties.
//  *
//  *         
//  *
//  *         \todo
//  *         \remarks           
//  *         \sa       NA
//  *         \date     04.01.2011
//  *         \author  Mauch
//  *
//  */
//class MatLinearGrating1D_DiffractionParams
//{
//public:
//	  /* destructor */
//    ~MatLinearGrating1D_DiffractionParams()
//    {
//		free(lambdaPtr);
//		free(diffOrdersPtr);
//		free(RTP01Ptr);
//		free(RTP10Ptr);
//		free(RTS01Ptr);
//		free(RTS10Ptr);
//	}
//	int nrWavelengths;
//	double *lambdaPtr;
//	short nrOrders; // number of orders of which efficiency data is present
//	short nrOrdersSim; // number of orders that are to be accounted for in the current simulation
//	short *diffOrdersPtr;
//	double *RTP01Ptr;
//	double *RTP10Ptr;
//	double *RTS01Ptr;
//	double *RTS10Ptr;
//	double g;
//	double3 diffAxis;
//};

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
  *\class   MaterialLinearGrating1D_DiffRays
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
class MaterialLinearGrating1D_DiffRays: public MaterialLinearGrating1D
{
	protected:

  public:
    /* standard constructor */
    MaterialLinearGrating1D_DiffRays()
	{
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_LINEARGRATING1D_DIFFRAYS );
//		this->setPathToPtx(path_to_ptx);
	}
    MaterialLinearGrating1D_DiffRays(int nrWave, int nrOrd )
	{
		/* set ptx path for OptiX calculations */
//		path_to_ptx[512];
		sprintf( path_to_ptx, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, PATH_TO_HIT_LINEARGRATING1D_DIFFRAYS );
//		this->setPathToPtx(path_to_ptx);

	}
	/* destrcuctor */
	~MaterialLinearGrating1D_DiffRays()
	{
//		if (diffractionParamsPtr != NULL)
//		{
//			delete this->diffractionParamsPtr;
//			this->diffractionParamsPtr=NULL;
//		}
//		if (this->glassDispersionParamsPtr!=NULL)
//			delete glassDispersionParamsPtr;
//		delete path_to_ptx;
	}
 //   MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	//MaterialError updateOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, SimParams simParams, double lambda);
	//MaterialError createCPUSimInstance(double lambda);
	//MaterialError updateCPUSimInstance(double lambda);

	void hit(diffRayStruct &ray, Mat_DiffRays_hitParams hitParams, double t_hit, int geometryID);
};

#endif


