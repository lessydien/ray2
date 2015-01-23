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

/**\file PhaseSpaceField.cpp
* \brief PhaseSpace representation of light field
* 
*           
* \author Mauch
*/

#include "PhaseSpaceField.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"
#include "inputOutput.h"
//#include "matrix.h"
#ifdef _MATSUPPORT
	#include "mat.h"
#endif

double* PhaseSpaceField::getPhaseSpacePtr()
{
	return this->PSptr;
};

//complex<double>* PhaseSpaceField::getComplexAmplPtr()
//{
//	return this->Uptr;
//};

phaseSpaceParams* PhaseSpaceField::getParamsPtr()
{
	return this->paramsPtr;
};

/**
 * \detail write2TextFile
 *
 * saves the field to a textfile format
 *
 * \param[in] FILE* hfile
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PhaseSpaceField::write2TextFile(char* filename, detParams &oDetParams)
{
	char t_filename[512];
	sprintf(t_filename, "%s%sPhaseSpaceField.txt", filename, PATH_SEPARATOR);

	FILE* hFileOut;
	hFileOut = fopen( t_filename, "w" ) ;
	if (!hFileOut)
	{
		std::cout << "error in PhaseSpaceField.write2TextFile(): could not open output file: " << filename << "...\n";
		return FIELD_ERR;
	}
	if ( IO_NO_ERR != writePhaseSpaceField2File(hFileOut, this) )
	{
		std::cout << "error in PhaseSpaceField.write2TextFile(): writePhaseSpaceFIeld2File() returned an error" << "...\n";
		return FIELD_ERR;
	}
	fclose(hFileOut);
	return FIELD_NO_ERR;
};

/**
 * \detail write2MatFile
 *
 * saves the field to a mat file
 *
 * \param[in] char* filename
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError PhaseSpaceField::write2MatFile(char* filename, detParams &oDetParams)
{
#ifdef _MATSUPPORT
	char t_filename[512];
	sprintf(t_filename, "%s%sPhaseSpaceField.mat", filename, PATH_SEPARATOR);
	MATFile *pmat;
	pmat = matOpen(t_filename, "w");
	if (pmat == NULL) 
	{
		std::cout << "error in PhaseSpaceField.write2MatFile(): could not open the mat file" << "...\n";
		return(FIELD_NO_ERR);
	}

	mxArray *mat_PS = NULL, *mat_lambda=NULL, *mat_MTransform=NULL, *mat_nrPixels=NULL, *mat_nrPixels_dir=NULL;
	mxArray *mat_scale=NULL, *mat_scale_dir=NULL;//, *mat_unitLambda=NULL, *mat_units=NULL;
	/* 
	 * Create variables from our data
	 */
	mwSize *dims=(mwSize*)malloc(4*sizeof(mwSize));
	dims[0]=this->paramsPtr->nrPixels.x;
	dims[1]=this->paramsPtr->nrPixels.y;
	dims[2]=this->paramsPtr->nrPixels_PhaseSpace.x;
	dims[3]=this->paramsPtr->nrPixels_PhaseSpace.y;
	mat_PS = mxCreateNumericArray(4, dims, mxDOUBLE_CLASS, mxREAL);
	memcpy((char *) mxGetPr(mat_PS), (char *) this->getPhaseSpacePtr(), this->paramsPtr->nrPixels.x*this->paramsPtr->nrPixels.y*this->paramsPtr->nrPixels_PhaseSpace.x*this->paramsPtr->nrPixels_PhaseSpace.y*sizeof(double));
	mat_lambda = mxCreateDoubleScalar(this->paramsPtr->lambda);
	mat_MTransform = mxCreateDoubleMatrix(4, 4, mxREAL);
	memcpy((char *) mxGetPr(mat_MTransform), (char *) &(this->paramsPtr->MTransform), 16*sizeof(double));
	mat_nrPixels = mxCreateDoubleMatrix(3, 1, mxREAL);
	double3 nrPixels;
	nrPixels.x=(double)(this->paramsPtr->nrPixels.x);
	nrPixels.y=(double)(this->paramsPtr->nrPixels.y);
	nrPixels.z=(double)(this->paramsPtr->nrPixels.z);
	memcpy((char *) mxGetPr(mat_nrPixels), (char *) &(nrPixels), 3*sizeof(double));
	mat_scale = mxCreateDoubleMatrix(3, 1, mxREAL);
	memcpy((char *) mxGetPr(mat_scale), (char *) &(this->paramsPtr->scale), 3*sizeof(double));
	mat_scale_dir = mxCreateDoubleMatrix(2, 1, mxREAL);
	memcpy((char *) mxGetPr(mat_scale_dir), (char *) &(this->paramsPtr->scale_dir), 2*sizeof(double));
	mat_nrPixels_dir = mxCreateDoubleMatrix(2, 1, mxREAL);
	double2 nrPixels_dir;
	nrPixels_dir.x=(double)(this->paramsPtr->nrPixels_PhaseSpace.x);
	nrPixels_dir.y=(double)(this->paramsPtr->nrPixels_PhaseSpace.y);
	memcpy((char *) mxGetPr(mat_nrPixels_dir), (char *) &(nrPixels_dir), 2*sizeof(double));

	/*
	 * Place the variables into the MATLAB workspace
	 */
	matPutVariable(pmat, "PS", mat_PS);
	matPutVariable(pmat, "lambda", mat_lambda);
	matPutVariable(pmat, "MTransform", mat_MTransform);
	matPutVariable(pmat, "nrPixel", mat_nrPixels);
	matPutVariable(pmat, "nrPixel_dir", mat_nrPixels_dir);
	matPutVariable(pmat, "scale", mat_scale);
	matPutVariable(pmat, "scale_dir", mat_scale_dir);
//	engPutVariable(oMatInterface.getEnginePtr(), "unitLambda", this->paramsPtr->unitLambda);
//	engPutVariable(oMatInterface.getEnginePtr(), "units", this->paramsPtr->units);

	/* create a struct containing all the information of the PhaseSpaceField */
	//int result=engEvalString(oMatInterface.getEnginePtr(), "PhaseSpaceField=struct");
	//result=engEvalString(oMatInterface.getEnginePtr(), "PhaseSpaceField.params=struct");
	//result=engEvalString(oMatInterface.getEnginePtr(), "PhaseSpaceField.params.lambda=lambda");
	//result=engEvalString(oMatInterface.getEnginePtr(), "PhaseSpaceField.params.MTransform=MTransform");
	//result=engEvalString(oMatInterface.getEnginePtr(), "PhaseSpaceField.params.nrPixel=nrPixel");
	//result=engEvalString(oMatInterface.getEnginePtr(), "PhaseSpaceField.params.scale=scale");
	//result=engEvalString(oMatInterface.getEnginePtr(), "PhaseSpaceField.I=I");
	///* save the struct into a .mat file */
	//char saveCommand[564];
	//sprintf(saveCommand, "save %s PhaseSpaceField;", t_filename);
	//result=engEvalString(oMatInterface.getEnginePtr(), saveCommand);

	/* plot results */
	//result=engEvalString(oMatInterface.getEnginePtr(), "x=-(PhaseSpaceField.params.nrPixel(1,1)-1)/2*PhaseSpaceField.params.scale(1,1):PhaseSpaceField.params.scale(1,1):(PhaseSpaceField.params.nrPixel(1,1)-1)/2*PhaseSpaceField.params.scale(1,1);");
	//result=engEvalString(oMatInterface.getEnginePtr(), "x=x+PhaseSpaceField.params.MTransform(4,1);");
	//result=engEvalString(oMatInterface.getEnginePtr(), "y=-(PhaseSpaceField.params.nrPixel(2,1)-1)/2*PhaseSpaceField.params.scale(2,1):PhaseSpaceField.params.scale(2,1):(PhaseSpaceField.params.nrPixel(2,1)-1)/2*PhaseSpaceField.params.scale(2,1);");
	//result=engEvalString(oMatInterface.getEnginePtr(), "y=y+PhaseSpaceField.params.MTransform(4,2);");
	//result=engEvalString(oMatInterface.getEnginePtr(), "figure; imagesc(x,y,PhaseSpaceField.I'); grid; xlabel('x [mm]'); ylabel('y [mm]'); title('image')");
	//result=engEvalString(oMatInterface.getEnginePtr(), "line=sum(PhaseSpaceField.I,1);");
	//result=engEvalString(oMatInterface.getEnginePtr(), "figure; plot(y,line); grid; xlabel('y [mm]'); ylabel('counts'); title('line')");
	/*
	 * We're done! Free memory, close MATLAB engine and exit.
	 */
	mxDestroyArray(mat_PS);
	mxDestroyArray(mat_lambda);
	mxDestroyArray(mat_MTransform);
	mxDestroyArray(mat_nrPixels);
	mxDestroyArray(mat_nrPixels_dir);
	mxDestroyArray(mat_scale);
	mxDestroyArray(mat_scale_dir);

	if (matClose(pmat) != 0) 
	{
		std::cout << "error in PhaseSpaceField.write2MatFile(): could not close the mat file" << "...\n";
		return(FIELD_NO_ERR);
	}
#else
	std::cout << "error in ScalarLightField.write2MatFile(): matlab not supported" << "...\n";
	return FIELD_ERR;
#endif
	return FIELD_NO_ERR;
};


