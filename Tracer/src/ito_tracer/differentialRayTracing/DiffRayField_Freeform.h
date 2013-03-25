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

/**\file DiffRayField_Freeform.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef DIFFRAYFIELD_FREEFORM_H
  #define DIFFRAYFIELD_FREEFORM_H

#include <optix.h>
#include "../rayData.h"
#include "stdlib.h"
#include "../DiffRayField.h"
#include "../Interpolator.h"
#include <ctime>
#include "../pugixml.hpp"

#define DIFFRAYFIELD_FREEFORM_PATHTOPTX "ITO-MacroSim_generated_rayGenerationDiffRayField_Freeform.cu.ptx"

///* declare class */
///**
//  *\class   rayFieldParams
//  *\ingroup Field
//  *\brief   params of a ray field
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
//class diffRayFieldParams : public diffRayFieldParams
//{
//public:
//	//long2 nrRayDirections;
//	double epsilon; //!> short distance that diff rays moved from cuastic
//};

/* declare class */
/**
  *\class   DiffRayField_Freeform
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
class DiffRayField_Freeform : public DiffRayField
{
  protected:
	//double3 oldPosition;
	//diffRayStruct* rayList;

	//fieldError write2TextFile(char* filename, detParams &oDetParams);
	//fieldError initGPUSubset(RTcontext &context, RTbuffer &seed_buffer_obj);
	fieldError initCPUSubset();
	void spline(double *x, double *y, const unsigned int width, const double yp1,  const double ypn, double *y2);
	void splint(double *xa, double *ya, double *y2a, const unsigned int width, const double x,  double &y);
	void splin2(double *x1a, double *x2a, const unsigned int width, const unsigned int height, double *ya, double *y2a, const double x1, const double x2, double &y);
	void splie2(double *x1a, double *x2a, double *ya, double *y2a, const unsigned int width, const unsigned int height);

	RTbuffer freeForm_buffer_obj;
	double* freeForm_buffer_CPU;
	RTbuffer freeForm_y2a_buffer_obj;
	double* freeForm_y2a_buffer_CPU;
	RTbuffer freeForm_x1a_buffer_obj;
	double* freeForm_x1a_buffer_CPU;
	RTbuffer freeForm_x2a_buffer_obj;
	double* freeForm_x2a_buffer_CPU;

	Interpolator *oInterpPtr;
	
	//diffRayFieldParams *rayParamsPtr;

  public:
    /* standard constructor */
    DiffRayField_Freeform()
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, DIFFRAYFIELD_FREEFORM_PATHTOPTX );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator

		rayList = NULL;
		rayListLength=0;
		materialList=NULL;
		materialListLength=0;
		rayParamsPtr=new diffRayFieldParams();
		freeForm_buffer_CPU=NULL;
		freeForm_y2a_buffer_CPU=NULL;
		freeForm_x1a_buffer_CPU=NULL;
		freeForm_x2a_buffer_CPU=NULL;

		oInterpPtr=new Interpolator();
		
	}
    /* Konstruktor */
    DiffRayField_Freeform(unsigned long long length)
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", SAMPLES_PTX_DIR, DIFFRAYFIELD_FREEFORM_PATHTOPTX );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator
		rayList = (diffRayStruct*) malloc(length*sizeof(diffRayStruct));
		rayListLength = length;
		materialList=NULL;
		materialListLength=0;
		rayParamsPtr=new diffRayFieldParams();
		freeForm_y2a_buffer_CPU=NULL;
		freeForm_x1a_buffer_CPU=NULL;
		freeForm_x2a_buffer_CPU=NULL;

		oInterpPtr=new Interpolator();
	}
	/* Destruktor */
	~DiffRayField_Freeform()
	{
	  if ( materialList != NULL)
	  {
		  int i;
		  for (i=0;i<materialListLength;i++)
		  {
			  if (materialList[i]!=NULL)
			  {
				  delete materialList[i];
			  }
		  }
		  materialList = NULL;
	  }
	  if ( rayList != NULL )
	  {
		delete rayList;
		rayList = NULL;
	  }
	  if ( rayParamsPtr != NULL )
	  {
		delete rayParamsPtr;
		rayParamsPtr=NULL;
	  }
	  if (freeForm_y2a_buffer_CPU != NULL)
	  {
		  delete freeForm_y2a_buffer_CPU;
		  freeForm_y2a_buffer_CPU=NULL;
	  }
	  if (freeForm_x1a_buffer_CPU != NULL)
	  {
		  delete freeForm_x1a_buffer_CPU;
		  freeForm_x1a_buffer_CPU=NULL;
	  }
	  if (freeForm_x2a_buffer_CPU != NULL)
	  {
		  delete freeForm_x2a_buffer_CPU;
		  freeForm_x2a_buffer_CPU=NULL;
	  }
	  if (oInterpPtr != NULL)
	  {
		  delete oInterpPtr;
		  oInterpPtr=NULL;
	  }


	}
	fieldError createCPUSimInstance();

    fieldError createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj);
};

#endif

