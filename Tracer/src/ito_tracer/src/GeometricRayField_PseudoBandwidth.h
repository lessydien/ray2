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

/**\file GeometricRayField_PseudoBandwidth.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef GEOMRAYFIELD_PSEUDOBANDWIDTH_H
  #define GEOMRAYFIELD_PSEUDOBANDWIDTH_H

#include <optix.h>
#include "rayData.h"
#include "stdlib.h"
#include "GeometricRayField.h"
#include "wavefrontIn.h"
#include "inputOutput.h"
#include <ctime>
#include "pugixml.hpp"


#define GEOMRAYFIELD_PSEUDOBANDWIDTH_PATHTOPTX "ITO-MacroSim_generated_rayGeneration_pseudoBandwidth.cu.ptx"

/* declare class */
/**
  *\class   GeometricRayField_PseudoBandwidth
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
class GeometricRayField_PseudoBandwidth : public GeometricRayField
{
  protected:
	
  public:
    /* standard constructor */
    GeometricRayField_PseudoBandwidth() :
		GeometricRayField()
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, GEOMRAYFIELD_PSEUDOBANDWIDTH_PATHTOPTX );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator
		rayList = NULL;
		rayListLength=0;
		materialList=NULL;
		materialListLength=0;
		rayParamsPtr=new rayFieldParams();
	}
    /* Konstruktor */
    GeometricRayField_PseudoBandwidth(unsigned long long length) :
		GeometricRayField(length)
	{
		sprintf( path_to_ptx_rayGeneration, "%s" PATH_SEPARATOR "%s", PATH_TO_PTX, GEOMRAYFIELD_PATHTOPTX );		
		// init random seed
		int seed = (int)time(0);            // random seed
		RandomInit(seed, x); // seed random generator
		rayList = (rayStruct*) malloc(length*sizeof(rayStruct));
		rayListLength = length;
		materialList=NULL;//new Material*[1];
		materialListLength=0;
		rayParamsPtr=new rayFieldParams();
	}
	/* Destruktor */
	~GeometricRayField_PseudoBandwidth()
	{
		if (materialList != NULL)
		{
			for (int i=0;i<materialListLength;i++)
			{
				if (materialList[i] != NULL)
				{
					delete materialList[i];
					materialList[i]=NULL;
				}
			}
			delete materialList;
			materialList=NULL;
		}
		if (rayList!=NULL)
		{
			delete rayList;
			rayList = NULL;
		}
		if (rayParamsPtr != NULL)
		{
			delete rayParamsPtr;
			rayParamsPtr=NULL;
		}
	}
					
	fieldError convert2Intensity(Field* imagePtr, detParams &oDetParams);
	fieldError convert2ScalarField(Field* imagePtr, detParams &oDetParams);
	fieldError convert2VecField(Field* imagePtr, detParams &oDetParams);

	fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec);
};

#endif

