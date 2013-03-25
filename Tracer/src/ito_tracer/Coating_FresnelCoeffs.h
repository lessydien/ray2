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

/**\file Coating_FresnelCoeffs.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef COATING_FRESNELCOEFFS_H
  #define COATING_FRESNELCOEFFS_H

#include <optix.h>
#include "Coating.h"
#include <optix_math.h>
#include "rayData.h"
#include "GlobalConstants.h"
#include "Coating_FresnelCoeffs_hit.h"
#include <stdio.h>
#include "MaterialRefracting.h"
//#include "CoatingLib.h"

#define PATH_TO_HIT_COAT_FRESNELCOEFFS "_CoatFresnelCoeffs"


/* declare class */
/**
  *\class   Coating_FresnelCoeffs_FullParams
  *\ingroup Coating
  *\brief   base class of full set of params of Coating_FresnelCoeffs
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
class Coating_FresnelCoeffs_FullParams: public Coating_FullParams
{
	public:
		Coating_FresnelCoeffs_FullParams()
		{
			type=CT_FRESNELCOEFFS;
			n1=0;
			n2=0;
			glassDispersionParamsPtr=NULL;
			immersionDispersionParamsPtr=NULL;
		}
		~Coating_FresnelCoeffs_FullParams()
		{
			if (glassDispersionParamsPtr!=NULL)
			{
				delete glassDispersionParamsPtr;
				glassDispersionParamsPtr=NULL;
			}
			if (immersionDispersionParamsPtr!=NULL)
			{
				delete immersionDispersionParamsPtr;
				immersionDispersionParamsPtr=NULL;
			}
		}
		MatRefracting_DispersionParams *glassDispersionParamsPtr;
		MatRefracting_DispersionParams *immersionDispersionParamsPtr;
		double n1; // refractive index
		double n2;
};

/* declare class */
/**
  *\class   Coating_FresnelCoeffs
  *\ingroup Coating
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
class Coating_FresnelCoeffs: public Coating
{
  protected:
    char* path_to_ptx;
	Coating_FresnelCoeffs_FullParams *fullParamsPtr;
	Coating_FresnelCoeffs_ReducedParams *reducedParamsPtr;
	//CoatingError calcCoatingCoeffs(double lambda, double3 normal, double3 direction);
		
  public:
	 Coating_FresnelCoeffs() // standard constructor
	 {
		 this->fullParamsPtr=new Coating_FresnelCoeffs_FullParams(); 
		 this->fullParamsPtr->type = CT_FRESNELCOEFFS;
		 this->reducedParamsPtr=new Coating_FresnelCoeffs_ReducedParams();
		 path_to_ptx=(char*)malloc(512*sizeof(char));
		 sprintf( path_to_ptx, "%s", PATH_TO_HIT_COAT_FRESNELCOEFFS );

	 }
	 ~Coating_FresnelCoeffs() // destructor
	 {
		 delete this->fullParamsPtr;
		 delete this->reducedParamsPtr;
		 delete path_to_ptx;
	 }

    CoatingError calcRefrIndices(double lambda);
	void setPathToPtx(char* path);
	CoatingError setFullParams(Coating_FresnelCoeffs_FullParams* ptrIn);
	Coating_FresnelCoeffs_FullParams* getFullParams(void);
	Coating_FresnelCoeffs_ReducedParams* getReducedParams(void);
	char* getPathToPtx(void);
//    virtual CoatingError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	CoatingError createCPUSimInstance(double lambda);
	CoatingError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	bool hit(rayStruct &ray, Mat_hitParams hitParams);
	bool hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	CoatingError processParseResults(MaterialParseParamStruct &parseResults_Mat, parseGlassResultStruct* parseResults_GlassPtr, parseGlassResultStruct* parseResults_ImmPtr);
};

#endif

