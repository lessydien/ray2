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

/**\file Field.h
* \brief 
* 
*           
* \author Mauch
*/

#ifndef FIELDSTACK_H
  #define FIELDSTACK_H

#include <optix.h>
#include "stdlib.h"
//#include "complex.h"
#include "macrosim_types.h"
//#include "optix_math.h"
#include "Field.h"

typedef enum 
{
  FIELDSTACK_NO_ERR,
  FIELDSTACK_ERR
} fieldStackError;

/* declare class */
/**
  *\class   fieldParams
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
class fieldStackParams : public fieldParams
{
public:
	double *runningParamPtr;
	long long runningParamLength;

    /* standard constructor */
    fieldStackParams()
	{
		this->runningParamPtr=NULL;
		this->runningParamLength=0;
	}
	/* destructor */
    ~fieldStackParams()
	{
		delete this->runningParamPtr;
		this->runningParamLength=0;
	}
};

/* declare class */
/**
  *\class   FieldStack
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
class FieldStack : public Field
{
  protected:

  public:
    /* standard constructor */
    FieldStack()
	{


	}
	/* Destruktor */
	~FieldStack()
	{

	}
	
	virtual fieldStackError addField(Field *fieldPtr, double runningParamIn, long long index);
	virtual Field* getField(long long index);

};

#endif

