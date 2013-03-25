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

/**\file Coating_DispersiveNumCoeffs.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef COATING_DISPERSIVENUMCOEFFS_H
  #define COATING_DISPERSIVENUMCOEFFS_H

#include <optix.h>
#include "Coating_NumCoeffs.h"
#include <optix_math.h>
#include "rayData.h"
#include "GlobalConstants.h"
#include "Coating_DispersiveNumCoeffs_hit.h"
#include <stdio.h>
//#include "CoatingLib.h"

#define PATH_TO_HIT_COAT_DISPNUMCOEFFS "_CoatDispersiveNumCoeffs"

/* declare class */
/**
  *\class   Coating_DispersiveNumCoeffs_FullParams
  *\ingroup Coating
  *\brief   base class of full set of params of Coating_DispersiveNumCoeffs
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
class Coating_DispersiveNumCoeffs_FullParams: public Coating_FullParams
{
	public:
		Coating_DispersiveNumCoeffs_FullParams()
		{
			type=CT_DISPNUMCOEFFS;
			a_r=0;
			c_r=0;
			a_t=0;
			c_t=0;
		}
		~Coating_DispersiveNumCoeffs_FullParams()
		{
		}
		double a_r;
		double c_r;
		double a_t;
		double c_t;
};

/* declare class */
/**
  *\class   Coating_DispersiveNumCoeffs
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
class Coating_DispersiveNumCoeffs: public Coating
{
  protected:
    char* path_to_ptx;
	Coating_DispersiveNumCoeffs_FullParams *fullParamsPtr;
	Coating_DispersiveNumCoeffs_ReducedParams *reducedParamsPtr;
	//CoatingError calcCoatingCoeffs(double lambda, double3 normal, double3 direction);
		
  public:
	 Coating_DispersiveNumCoeffs() // standard constructor
	 {
		 this->fullParamsPtr=new Coating_DispersiveNumCoeffs_FullParams(); 
		 this->fullParamsPtr->type = CT_NUMCOEFFS;
		 this->fullParamsPtr->a_t=0;
		 this->fullParamsPtr->c_t=1;
		 this->fullParamsPtr->a_r=0;
		 this->fullParamsPtr->c_t=0;
		 this->reducedParamsPtr=new Coating_DispersiveNumCoeffs_ReducedParams();
		 path_to_ptx=(char*)malloc(512*sizeof(char));
		 sprintf( path_to_ptx, "%s", PATH_TO_HIT_COAT_DISPNUMCOEFFS );

	 }
	 ~Coating_DispersiveNumCoeffs() // destructor
	 {
		 delete this->fullParamsPtr;
		 delete this->reducedParamsPtr;
		 delete path_to_ptx;
	 }

	void setPathToPtx(char* path);
	CoatingError setFullParams(Coating_DispersiveNumCoeffs_FullParams* ptrIn);
	Coating_DispersiveNumCoeffs_FullParams* getFullParams(void);
	Coating_DispersiveNumCoeffs_ReducedParams* getReducedParams(void);
	char* getPathToPtx(void);
//    virtual CoatingError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index, simMode mode, double lambda);
	CoatingError createCPUSimInstance(double lambda);
	CoatingError setParams2Program( RTcontext context, RTprogram *closest_hit_programPtr, RTvariable *l_scatterParamsPtr);
	bool hit(rayStruct &ray, Mat_hitParams hitParams);
	bool hit(gaussBeamRayStruct &ray, gaussBeam_geometricNormal normal);
	CoatingError processParseResults(MaterialParseParamStruct &parseResults_Mat);
};

#endif

