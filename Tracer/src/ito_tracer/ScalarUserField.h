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

/**\file ScalarUserField.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef SCALARUSERFIELD_H
  #define SCALARUSERFIELD_H

#include <optix.h>
#include "stdlib.h"
//#include "complex.h"
#include <complex>
#include "ScalarLightField.h"
//#include "my_vector_types.h"
#include <optix_math.h>
#include "pugixml.hpp"

/* declare class */
/**
  *\class   ScalarUserField
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
class ScalarUserField: public ScalarLightField
{
  protected:

  public:
    /* standard constructor */
    ScalarUserField() :
		ScalarLightField()
	{
	}
    /* Konstruktor */
    ScalarUserField(scalarFieldParams paramsIn) :
		ScalarLightField(paramsIn)
	{
	}
	/* Destruktor */
	~ScalarUserField()
	{
	  if (paramsPtr != NULL)
	  {
		delete paramsPtr;
		paramsPtr=NULL;
	  }
	}

	fieldError parseXml(pugi::xml_node &field, vector<Field*> &fieldVec);
};

#endif

