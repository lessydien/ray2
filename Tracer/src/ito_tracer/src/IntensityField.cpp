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

/**\file IntensityField.cpp
* \brief Intensity representation of light field
* 
*           
* \author Mauch
*/

#include "IntensityField.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"
#include "inputOutput.h"
#ifdef _MATSUPPORT
	#include "mat.h"
#endif

double* IntensityField::getIntensityPtr()
{
	return this->Iptr;
};

complex<double>* IntensityField::getComplexAmplPtr()
{
	return this->Uptr;
};

fieldParams* IntensityField::getParamsPtr()
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
fieldError IntensityField::write2TextFile(char* filename, detParams &oDetParams)
{
	char t_filename[512];
	sprintf(t_filename, "%s%s%s", filename, PATH_SEPARATOR, oDetParams.filenamePtr);

	FILE* hFileOut;
	hFileOut = fopen( t_filename, "w" ) ;
//	hFileOut = fopen( oDetParams.filenamePtr, "w" );

	if (!hFileOut)
	{
		std::cout << "error in IntensityField.write2TextFile(): could not open output file: " << filename << std::endl;
		return FIELD_ERR;
	}
	if ( IO_NO_ERR != writeIntensityField2File(hFileOut, this) )
	{
		std::cout << "error in IntensityField.write2TextFile(): writeIntensityFIeld2File() returned an error" << std::endl;
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
fieldError IntensityField::write2MatFile(char* filename, detParams &oDetParams)
{
#ifdef _MATSUPPORT
	char t_filename[512];
	sprintf(t_filename, "%s%s%s", filename, PATH_SEPARATOR, oDetParams.filenamePtr);
	MATFile *pmat;
	pmat = matOpen(t_filename, "w");
//	pmat = matOpen( oDetParams.filenamePtr, "w" );
	if (pmat == NULL) 
	{
		std::cout << "error in IntensityField.write2MatFile(): could not open the mat file" << std::endl;
		return(FIELD_NO_ERR);
	}

	mxArray *mat_I = NULL, *mat_lambda=NULL, *mat_MTransform=NULL, *mat_nrPixels=NULL;
	mxArray *mat_scale=NULL;//, *mat_unitLambda=NULL, *mat_units=NULL;
	/* 
	 * Create variables from our data
	 */
	mat_I = mxCreateDoubleMatrix(this->paramsPtr->nrPixels.x, this->paramsPtr->nrPixels.y, mxREAL);
	memcpy((char *) mxGetPr(mat_I), (char *) this->getIntensityPtr(), this->paramsPtr->nrPixels.x*this->paramsPtr->nrPixels.y*sizeof(double));
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

	/*
	 * Place the variables into the MATLAB workspace
	 */
	matPutVariable(pmat, "I", mat_I);
	matPutVariable(pmat, "lambda", mat_lambda);
	matPutVariable(pmat, "MTransform", mat_MTransform);
	matPutVariable(pmat, "nrPixel", mat_nrPixels);
	matPutVariable(pmat, "scale", mat_scale);
//	engPutVariable(oMatInterface.getEnginePtr(), "unitLambda", this->paramsPtr->unitLambda);
//	engPutVariable(oMatInterface.getEnginePtr(), "units", this->paramsPtr->units);

	/* create a struct containing all the information of the IntensityField */
	//int result=engEvalString(oMatInterface.getEnginePtr(), "IntensityField=struct");
	//result=engEvalString(oMatInterface.getEnginePtr(), "IntensityField.params=struct");
	//result=engEvalString(oMatInterface.getEnginePtr(), "IntensityField.params.lambda=lambda");
	//result=engEvalString(oMatInterface.getEnginePtr(), "IntensityField.params.MTransform=MTransform");
	//result=engEvalString(oMatInterface.getEnginePtr(), "IntensityField.params.nrPixel=nrPixel");
	//result=engEvalString(oMatInterface.getEnginePtr(), "IntensityField.params.scale=scale");
	//result=engEvalString(oMatInterface.getEnginePtr(), "IntensityField.I=I");
	///* save the struct into a .mat file */
	//char saveCommand[564];
	//sprintf(saveCommand, "save %s IntensityField;", t_filename);
	//result=engEvalString(oMatInterface.getEnginePtr(), saveCommand);

	/* plot results */
	//result=engEvalString(oMatInterface.getEnginePtr(), "x=-(IntensityField.params.nrPixel(1,1)-1)/2*IntensityField.params.scale(1,1):IntensityField.params.scale(1,1):(IntensityField.params.nrPixel(1,1)-1)/2*IntensityField.params.scale(1,1);");
	//result=engEvalString(oMatInterface.getEnginePtr(), "x=x+IntensityField.params.MTransform(4,1);");
	//result=engEvalString(oMatInterface.getEnginePtr(), "y=-(IntensityField.params.nrPixel(2,1)-1)/2*IntensityField.params.scale(2,1):IntensityField.params.scale(2,1):(IntensityField.params.nrPixel(2,1)-1)/2*IntensityField.params.scale(2,1);");
	//result=engEvalString(oMatInterface.getEnginePtr(), "y=y+IntensityField.params.MTransform(4,2);");
	//result=engEvalString(oMatInterface.getEnginePtr(), "figure; imagesc(x,y,IntensityField.I'); grid; xlabel('x [mm]'); ylabel('y [mm]'); title('image')");
	//result=engEvalString(oMatInterface.getEnginePtr(), "line=sum(IntensityField.I,1);");
	//result=engEvalString(oMatInterface.getEnginePtr(), "figure; plot(y,line); grid; xlabel('y [mm]'); ylabel('counts'); title('line')");
	/*
	 * We're done! Free memory, close MATLAB engine and exit.
	 */
	mxDestroyArray(mat_I);
	mxDestroyArray(mat_lambda);
	mxDestroyArray(mat_MTransform);
	mxDestroyArray(mat_nrPixels);
	mxDestroyArray(mat_scale);

	if (matClose(pmat) != 0) 
	{
		std::cout << "error in ScalarLightField.write2MatFile(): could not close the mat file" << std::endl;
		return(FIELD_NO_ERR);
	}
#else
	std::cout << "error in ScalarLightField.write2MatFile(): matlab not supported" << std::endl;
	return FIELD_ERR;
#endif
	return FIELD_NO_ERR;
};

/**
 * \detail convert2ItomObject 
 *
 * \param[in] void** dataPtrPtr, ItomFieldParams** paramsOut
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mau
 */
fieldError IntensityField::convert2ItomObject(void** dataPtrPtr, ItomFieldParams* paramsOut)
{
	// copy paramters
	this->convertFieldParams2ItomFieldParams(paramsOut);
	paramsOut->type=INTFIELD;

	unsigned long long l_size=this->getParamsPtr()->nrPixels.x*this->getParamsPtr()->nrPixels.y*this->getParamsPtr()->nrPixels.z*sizeof(double);
	(*dataPtrPtr)=malloc(l_size);
	memcpy((*dataPtrPtr), this->getIntensityPtr(), l_size);

	return FIELD_NO_ERR;
};

/**
 * \detail parseXml
 *
 * \param[in] pugi::xml_node &field, vector<Field*> &fieldVec
 * 
 * \return void
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError  IntensityField::parseXml(pugi::xml_node &det, vector<Field*> &fieldVec, SimParams simParams)
{
	// call base class function
	if (FIELD_NO_ERR != Field::parseXml(det, fieldVec, simParams))
	{
		std::cout << "error in IntensityField.parseXml(): Field.parseXml()  returned an error." << std::endl;
		return FIELD_ERR;
	}
	return FIELD_NO_ERR;
};
