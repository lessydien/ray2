/*
Copyright (C) 2012 ITO university stuttgart

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; If not, see <http://www.gnu.org/licenses/>.

*/

/**\file MatlabInterface.cpp
* \brief collection of functions to handle input and output to matlab
* 
*           
* \author Mauch
*/

#include "MatlabInterface.h"
#include <stdio.h>
#include <iostream>


//MatInterfaceError MatlabInterface::init(void)
//{
//	// check wether the engine 
//	/*
//	 * Start the MATLAB engine 
//	 */
//	if (!(matlabEnginePtr = engOpen(NULL))) 
//	{
//		std::cout << "error in MatlabInterface.init(): could not open matlab engine" << std::endl;
//		return MATINT_ERR;
//	}
//	return MATINT_NO_ERR;
//};
//
//Engine* MatlabInterface::getEnginePtr(void)
//{
//	return this->matlabEnginePtr;
//}
