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

/**\file inputOutput.cpp
* \brief collection of functions to handle input and output of lightfield data
* 
*           
* \author Mauch
*/

#include "inputOutput.h"
#include "math.h"
//#include "engine.h"
#include <stdio.h>
#include <iostream>

InputOutputError writeGeomRayData2File(FILE* hFile, rayStruct* prdPtr, long long int length, rayDataOutParams params)
{
	/* write CPU results to file */
	if( (hFile == NULL) )
	{
		std::cout << "error in writeGeomRayData2File(): invalid file handle" << "...\n";
		return IO_FILE_ERR;
	}
	else
	{
		if (params.reducedData)
		{
			for (int j=0; j<length; j++)
			{
				if ( (prdPtr[j].currentGeometryID==params.ID) || (params.ID==-1) )
				{
					// write the data in row major format, where width is the size of one row and height is the size of one coloumn
					// if the end of a row is reached append a line feed 
					fprintf(hFile, "%.20lf ;%.20lf ;%.20lf; ;%.20lf \n", prdPtr[j].position.x, prdPtr[j].position.y, prdPtr[j].position.z, prdPtr[j].flux);
				}
			}
		}
		else
		{
			for (int j=0; j<length; j++)
			{
				if ( (prdPtr[j].currentGeometryID==params.ID) || (params.ID==-1) )
				{
					// write the data in row major format, where width is the size of one row and height is the size of one coloumn
					// if the end of a row is reached append a line feed 
					fprintf(hFile, "%.20lf ;%.20lf ;%.20lf; %.20lf ;%.20lf ;%.20lf; %.20lf; %.20lf; %i; %i; %.20lf; \n", prdPtr[j].position.x, prdPtr[j].position.y, prdPtr[j].position.z, prdPtr[j].direction.x, prdPtr[j].direction.y, prdPtr[j].direction.z, prdPtr[j].flux, prdPtr[j].opl, prdPtr[j].currentGeometryID, prdPtr[j].depth, prdPtr[j].nImmersed);
				}
			}
		} // end if reducedData
	} // end if RunOnCPU
	return IO_NO_ERR;
};

InputOutputError writeScalarField2File(FILE* hFile, ScalarLightField *ptrLightField)
{
	if( hFile == NULL )
	{
		std::cout << "error in writeScalarField2File(): invalid file handle" << "...\n";
		return IO_FILE_ERR;
	}
	else
	{
		unsigned long long jy,jx;
		for (jy=0; jy<ptrLightField->getParamsPtr()->nrPixels.y; jy++)
		{
			jx=0;
			for (jx=0; jx<ptrLightField->getParamsPtr()->nrPixels.x; jx++)
			{
				// write the date in row major format, where width is the size of one row and height is the size of one coloumn
				// if the end of a row is reached append a line feed 
				if (jx+1 == ptrLightField->getParamsPtr()->nrPixels.x)
				{
					fprintf(hFile, " %.16e, %.16e,\n", ptrLightField->getPix(make_ulong2(jx,jy)).real(), ptrLightField->getPix(make_ulong2(jx,jy)).imag());
				}
				else
				{
					fprintf(hFile, " %.16e, %.16e,", ptrLightField->getPix(make_ulong2(jx,jy)).real(), ptrLightField->getPix(make_ulong2(jx,jy)).imag());
				}
			}
		}
	}
	return IO_NO_ERR;
};

InputOutputError writeIntensityField2File(FILE* hFile, IntensityField *ptrIntensityField)
{
	if( hFile == NULL )
	{
		std::cout << "error in writeIntensityField2File(): invalid file handle" << "...\n";
		return IO_FILE_ERR;
	}
	else
	{
		unsigned long long jy,jx;
		for (jy=0; jy<ptrIntensityField->getParamsPtr()->nrPixels.y; jy++)
		{
			jx=0;
			for (jx=0; jx<ptrIntensityField->getParamsPtr()->nrPixels.x; jx++)
			{
				// write the date in row major format, where width is the size of one row and height is the size of one coloumn
				// if the end of a row is reached append a line feed 
				if (jx+1 == ptrIntensityField->getParamsPtr()->nrPixels.x)
				{
					fprintf(hFile, " %.16e;\n", ptrIntensityField->getPix(make_ulong2(jx,jy)));
				}
				else
				{
					fprintf(hFile, " %.16e;", ptrIntensityField->getPix(make_ulong2(jx,jy)));
				}
			}
		}
	}
	return IO_NO_ERR;
};

InputOutputError writePhaseSpaceField2File(FILE* hFile, PhaseSpaceField *ptrPhaseSpaceField)
{
	if( hFile == NULL )
	{
		std::cout << "error in writeIntensityField2File(): invalid file handle" << "...\n";
		return IO_FILE_ERR;
	}
	else
	{
		//unsigned long long jy,jx;
		//for (jy=0; jy<ptrPhaseSpaceField->getParamsPtr()->nrPixels.y; jy++)
		//{
		//	jx=0;
		//	for (jx=0; jx<ptrPhaseSpaceField->getParamsPtr()->nrPixels.x; jx++)
		//	{
		//		// write the date in row major format, where width is the size of one row and height is the size of one coloumn
		//		// if the end of a row is reached append a line feed 
		//		if (jx+1 == ptrPhaseSpaceField->getParamsPtr()->nrPixels.x)
		//		{
		//			fprintf(hFile, " %.16e;\n", ptrPhaseSpaceField->getPix(make_ulong2(jx,jy)));
		//		}
		//		else
		//		{
		//			fprintf(hFile, " %.16e;", ptrPhaseSpaceField->getPix(make_ulong2(jx,jy)));
		//		}
		//	}
		//}
	}
	return IO_NO_ERR;
};

//InputOutputError loadMatlabEngine()
//{
//	Engine *ep;
//	/*
//	 * Start the MATLAB engine 
//	 */
//	if (!(ep = engOpen(NULL))) {
//		return IO_MATLAB_ERR;
//	}
//}
