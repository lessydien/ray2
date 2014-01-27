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

/**\file ScalarLightField.cpp
* \brief scalar representation of light field
* 
*           
* \author Mauch
*/

#include "ScalarLightField.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"
#ifdef _MATSUPPORT
	#include "mat.h"
#endif
#include "Parser_XML.h"
#include <ctime>


complex<double>* ScalarLightField::getFieldPtr()
{
	return this->U;
};

fieldParams* ScalarLightField::getParamsPtr()
{
	return (this->paramsPtr);
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
fieldError ScalarLightField::write2TextFile(char* filename, detParams &oDetParams)
{
	std::cout << "error in ScalarLightField.write2TextFile(): not implemented yet" << std::endl;
	return FIELD_ERR;
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
fieldError ScalarLightField::write2MatFile(char* filename, detParams &oDetParams)
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
//	MatlabInterface oMatInterface;
	mxArray *mat_U = NULL, *mat_lambda=NULL, *mat_MTransform=NULL, *mat_nrPixels=NULL;
	mxArray *mat_scale=NULL;//, *mat_unitLambda=NULL, *mat_units=NULL;
	/* 
	 * Create variables from our data
	 */
	mat_U = mxCreateDoubleMatrix(this->paramsPtr->nrPixels.x, this->paramsPtr->nrPixels.y,  mxCOMPLEX);
	double* Uimag = mxGetPi(mat_U);
	double* Ureal = mxGetPr(mat_U);
	for (unsigned long long ix=0;ix<this->paramsPtr->nrPixels.x;ix++)
	{
		for (unsigned long long iy=0;iy<this->paramsPtr->nrPixels.y;iy++)
		{
			//Ureal[ix+iy*this->paramsPtr->nrPixels.x]=d_real(this->U[ix+iy*this->paramsPtr->nrPixels.x]);
			Ureal[ix+iy*this->paramsPtr->nrPixels.x]=real(this->U[ix+iy*this->paramsPtr->nrPixels.x]);
			//Uimag[ix+iy*this->paramsPtr->nrPixels.x]=d_imag(this->U[ix+iy*this->paramsPtr->nrPixels.x]);
			Uimag[ix+iy*this->paramsPtr->nrPixels.x]=imag(this->U[ix+iy*this->paramsPtr->nrPixels.x]);
		}
	}
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
	matPutVariable(pmat, "U", mat_U);
	matPutVariable(pmat, "lambda", mat_lambda);
	matPutVariable(pmat, "MTransform", mat_MTransform);
	matPutVariable(pmat, "nrPixel", mat_nrPixels);
	matPutVariable(pmat, "scale", mat_scale);
//	engPutVariable(oMatInterface.getEnginePtr(), "unitLambda", this->paramsPtr->unitLambda);
//	engPutVariable(oMatInterface.getEnginePtr(), "units", this->paramsPtr->units);

	/* create a struct containing all the information of the IntensityField */
	//int result=engEvalString(oMatInterface.getEnginePtr(), "ScalarField=struct");
	//result=engEvalString(oMatInterface.getEnginePtr(), "ScalarField.params=struct");
	//result=engEvalString(oMatInterface.getEnginePtr(), "ScalarField.params.lambda=lambda");
	//result=engEvalString(oMatInterface.getEnginePtr(), "ScalarField.params.MTransform=MTransform");
	//result=engEvalString(oMatInterface.getEnginePtr(), "ScalarField.params.nrPixel=nrPixel");
	//result=engEvalString(oMatInterface.getEnginePtr(), "ScalarField.params.scale=scale");
	//result=engEvalString(oMatInterface.getEnginePtr(), "ScalarField.U=U");
	///* save the struct into a .mat file */
	//char saveCommand[564];
	//sprintf(saveCommand, "save %s ScalarField;", t_filename);
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
	mxDestroyArray(mat_U);
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
 * \detail initGPUSubset 
 *
 * \param[in] RTcontext &context
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarLightField::initGPUSubset(RTcontext &context)
{
	std::cout << "error in ScalarLightField.initGPUSubset(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail initCPUSubset 
 *
 * \param[in] 
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarLightField::initCPUSubset()
{
	std::cout << "error in ScalarLightField.initCPUSubset(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail createCPUSimInstance 
 *
 * \param[in] 
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarLightField::createCPUSimInstance()
{
	std::cout << "error in ScalarLightField.createCPUSimInstance(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail createOptixInstance 
 *
 * \param[in] RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarLightField::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	std::cout << "error in ScalarLightField.createOptixInstance(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail traceScene 
 *
 * \param[in] Group &oGroup
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarLightField::traceScene(Group &oGroup, bool RunOnCPU)
{
	double l_dz=10;
	double *l_px1=(double*)malloc(this->getParamsPtr()->nrPixels.x*sizeof(double));
	double *l_py1=(double*)malloc(this->getParamsPtr()->nrPixels.y*sizeof(double));
	for (unsigned int ix=0; ix<this->getParamsPtr()->nrPixels.x; ix++)
	{
		l_px1[ix]=-(this->getParamsPtr()->nrPixels.x/2*this->getParamsPtr()->scale.x-this->getParamsPtr()->scale.x/2)+ix*this->getParamsPtr()->scale.x;
	}
	for (unsigned int iy=0; iy<this->getParamsPtr()->nrPixels.y; iy++)
	{
		l_py1[iy]=-(this->getParamsPtr()->nrPixels.y/2*this->getParamsPtr()->scale.y-this->getParamsPtr()->scale.y/2)+iy*this->getParamsPtr()->scale.y;
	}

	double *l_px2, *l_py2;

	if (PROP_NO_ERR != fraunhofer(this->U, this->getParamsPtr()->nrPixels.x, this->getParamsPtr()->nrPixels.y, this->getParamsPtr()->lambda, l_px1, l_py1, l_dz, &l_px2, &l_py2))
	{
		std::cout << "error in ScalarLightField.traceScene(): fraunhofer() returned an error" << std::endl;
		return FIELD_ERR;
	}
	//std::cout <<"error in ScalarLightField.traceScene(): this has not yet been implemented for the given Field representation" << std::endl;
	return FIELD_NO_ERR;
}

/**
 * \detail convert2ScalarField 

 *
 * \param[in] Field* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarLightField::convert2ScalarField(Field* imagePtr, detParams &oDetParams)
{
	ScalarLightField* l_ScalarImagePtr=dynamic_cast<ScalarLightField*>(imagePtr);
	if (l_ScalarImagePtr == NULL)
	{
		std::cout << "error in ScalarLightField.convert2ScalarField(): imagePtr is not of type ScalarLightField" << std::endl;
		return FIELD_ERR;
	}

	if ( (imagePtr->getParamsPtr()->nrPixels.x != this->getParamsPtr()->nrPixels.x)
		|| (imagePtr->getParamsPtr()->nrPixels.y != this->getParamsPtr()->nrPixels.y)
		|| (imagePtr->getParamsPtr()->nrPixels.z != this->getParamsPtr()->nrPixels.z)
		|| (imagePtr->getParamsPtr()->scale.x != this->getParamsPtr()->scale.x)
		|| (imagePtr->getParamsPtr()->scale.y != this->getParamsPtr()->scale.y)
		|| (imagePtr->getParamsPtr()->scale.z != this->getParamsPtr()->scale.z) )
	{
		std::cout << "warning in ScalarLightField.convert2ScalarField(): detector image has different sampling than field. Resampling is necessary..." << std::endl;
		return FIELD_ERR;
	}
	else
	{
		for (unsigned long jx=0;jx<this->getParamsPtr()->nrPixels.x;jx++)
		{
			for (unsigned long jy=0;jy<this->getParamsPtr()->nrPixels.y;jy++)
			{
				for (unsigned long jz=0;jz<this->getParamsPtr()->nrPixels.z;jz++)
				{
					l_ScalarImagePtr->U[jx+jy*getParamsPtr()->nrPixels.x+jz*getParamsPtr()->nrPixels.y]=this->U[jx+jy*getParamsPtr()->nrPixels.x+jz*getParamsPtr()->nrPixels.y];
				}
			}
		}
//		memcpy(l_ScalarImagePtr->U=NULL, this->U, this->getParamsPtr()->nrPixels.x*this->getParamsPtr()->nrPixels.y*this->getParamsPtr()->nrPixels.z*sizeof(complex<double>));
	}

	return FIELD_NO_ERR;
};

/**
 * \detail convert2Intensity 

 *
 * \param[in] Field* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError ScalarLightField::convert2Intensity(Field* imagePtr, detParams &oDetParams)
{

	return FIELD_NO_ERR;
};

/**
 * \detail doSim
 *
 * \param[in] 
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError  ScalarLightField::doSim(Group &oGroup, simAssParams &params, bool &simDone)
{
	std::cout << "****************************************************** " << std::endl;
	std::cout << "starting subset.......... " << std::endl;
	std::cout << std::endl;
	/***********************************************
	/	trace ...
	/***********************************************/

	if (FIELD_NO_ERR != this->traceScene(oGroup, params.RunOnCPU) )
	{
		std::cout << "error in ScalarLightField.doSim(): ScalarLightField.traceScene() returned an error" << std::endl;
		return FIELD_ERR;
	}

	// signal end of simulation
	simDone=true;

	//if (simDone)
	//{
	//	if (!params.RunOnCPU)
	//	{
	//		// clean up
	//		if (!RT_CHECK_ERROR_NOEXIT( rtContextDestroy( context ), context ))
	//			return FIELD_ERR;
	//	}
	//}
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
fieldError  ScalarLightField::parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams)
{
	// call base class function
	if (FIELD_NO_ERR != Field::parseXml(field, fieldVec, simParams))
	{
		std::cout << "error in ScalarLightField.parseXml(): Field.parseXml()  returned an error." << std::endl;
		return FIELD_ERR;
	}
	this->paramsPtr->unitLambda=metric_mm;
	axesUnits l_units;
	l_units.x=metric_mm;
	l_units.y=metric_mm;
	l_units.z=metric_mm;
	this->paramsPtr->units=l_units;


	Parser_XML l_parser;
	scalarFieldParams *l_pParams=reinterpret_cast<scalarFieldParams*>(this->getParamsPtr());
	if (!l_parser.attrByNameToDouble(field, "amplMax", l_pParams->amplMax))
	{
		std::cout << "error in ScalarLightField.parseXml(): amplMax is not defined" << std::endl;
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToSLong(field, "numberOfPixels.x", this->getParamsPtr()->nrPixels.x))
	{
		std::cout << "error in ScalarLightField.parseXml(): numberOfPixels.x is not defined" << std::endl;
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToSLong(field, "numberOfPixels.y", this->getParamsPtr()->nrPixels.y))
	{
		std::cout << "error in ScalarLightField.parseXml(): numberOfPixels.y is not defined" << std::endl;
		return FIELD_ERR;
	}
	this->getParamsPtr()->nrPixels.z=1;

	double2 l_aprtHalfWidth;
	if (!l_parser.attrByNameToDouble(field, "apertureHalfWidth.x", l_aprtHalfWidth.x))
	{
		std::cout << "error in ScalarLightField.parseXml(): apertureHalfWidth.x is not defined" << std::endl;
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "apertureHalfWidth.y", l_aprtHalfWidth.y))
	{
		std::cout << "error in ScalarLightField.parseXml(): apertureHalfWidth.y is not defined" << std::endl;
		return FIELD_ERR;
	}
	this->getParamsPtr()->scale.x=2*l_aprtHalfWidth.x/this->getParamsPtr()->nrPixels.x;
	this->getParamsPtr()->scale.y=2*l_aprtHalfWidth.y/this->getParamsPtr()->nrPixels.y;
	this->getParamsPtr()->scale.z=0.01/this->getParamsPtr()->nrPixels.z;
	double3 l_tilt;
	double3 l_root;

	if (!l_parser.attrByNameToDouble(field, "root.x", l_root.x))
	{
		std::cout << "error in ScalarLightField.parseXml(): root.x is not defined" << std::endl;
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "root.y", l_root.y))
	{
		std::cout << "error in ScalarLightField.parseXml(): root.y is not defined" << std::endl;
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "root.z", l_root.z))
	{
		std::cout << "error in ScalarLightField.parseXml(): root.z is not defined" << std::endl;
		return FIELD_ERR;
	}
	if (!l_parser.attrByNameToDouble(field, "tilt.x", l_tilt.x))
	{
		std::cout << "error in ScalarLightField.parseXml(): tilt.x is not defined" << std::endl;
		return FIELD_ERR;
	}
	l_tilt.x=l_tilt.x/360*2*PI;
	if (!l_parser.attrByNameToDouble(field, "tilt.y", l_tilt.y))
	{
		std::cout << "error in ScalarLightField.parseXml(): tilt.y is not defined" << std::endl;
		return FIELD_ERR;
	}
	l_tilt.y=l_tilt.y/360*2*PI;
	if (!l_parser.attrByNameToDouble(field, "tilt.z", l_tilt.z))
	{
		std::cout << "error in ScalarLightField.parseXml(): tilt.z is not defined" << std::endl;
		return FIELD_ERR;
	}
	l_tilt.z=l_tilt.z/360*2*PI;

	this->getParamsPtr()->MTransform=createTransformationMatrix(l_tilt, l_root);
	return FIELD_NO_ERR;
};