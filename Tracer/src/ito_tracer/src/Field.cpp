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

/**\file Field.cpp
* \brief base class for all light field representations
* 
*           
* \author Mauch
*/

#include "Field.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "math.h"
#include "Parser_XML.h"

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
fieldError Field::write2TextFile(char* filename, detParams &oDetParams)
{
	std::cout << "error in Field.write2TextFile(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
}

/**
 * \detail write2MatFile
 *
 * saves the field to a mat file
 *
 * \param[in] char* filename, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::write2MatFile(char* filename, detParams &oDetParams)
{
	std::cout << "error in Field.write2MatFile(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail write2File
 *
 * saves the field to a file
 *
 * \param[in] char* filename, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::write2File(char* filename, detParams &oDetParams)
{
	fieldError err;
	switch (oDetParams.outFormat)
	{
	case DET_OUT_TEXT:
		err=write2TextFile(filename, oDetParams);
		break;
	case DET_OUT_MAT:
		err=write2MatFile(filename, oDetParams);
		break;
	default:
		std::cout << "error in Field.write2File(): outFormat " << oDetParams.outFormat << " not implemented yet" << std::endl;
		return FIELD_ERR;
	}
	return err;
	//std::cout << "error in Field.write2MatFile(): not defined for the given field representation" << std::endl;
	//return FIELD_ERR;
};

/**
 * \detail convert2Intensity 
 *
 * \param[in] IntensityField* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::convert2Intensity(Field* imagePtr, detParams &oDetParams)
{
	std::cout << "error in Field.convert2Intensity(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail convert2PhaseSpace 
 *
 * \param[in] IntensityField* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::convert2PhaseSpace(Field* imagePtr, detParams &oDetParams)
{
	std::cout << "error in Field.convert2PhaseSpace(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail convert2ScalarField 
 *
 * \param[in] ScalarLightField* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::convert2ScalarField(Field* imagePtr, detParams &oDetParams)
{
	std::cout << "error in Field.convert2Intensity(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail convert2VecField 
 *
 * \param[in] VectorLightField* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::convert2VecField(Field* imagePtr, detParams &oDetParams)
{
	std::cout << "error in Field.convert2Intensity(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail convert2ItomObject 
 *
 * \param[in] void** dataPtrPtr, ItomFieldParams** paramsOut
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::convert2ItomObject(void** dataPtrPtr, ItomFieldParams* paramsOut)
{
	std::cout << "error in Field.convert2ItomObject(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
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
fieldError Field::initGPUSubset(RTcontext &context)
{
	std::cout << "error in Field.initGPUSubset(): not defined for the given field representation" << std::endl;
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
fieldError Field::initCPUSubset()
{
	std::cout << "error in Field.initCPUSubset(): not defined for the given field representation" << std::endl;
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
fieldError Field::createCPUSimInstance()
{
	std::cout << "error in Field.createCPUSimInstance(): not defined for the given field representation" << std::endl;
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
fieldError Field::createOptixInstance(RTcontext &context, RTbuffer &output_buffer_obj, RTbuffer &seed_buffer_obj)
{
	std::cout << "error in Field.createOptixInstance(): not defined for the given field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail convertFieldParams2ItomFieldParams 
 *
 * \param[in] ItomFieldParams* paramsOut
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::convertFieldParams2ItomFieldParams(ItomFieldParams* paramsOut)
{
	// copy paramters
	paramsOut->lambda=this->getParamsPtr()->lambda;
	paramsOut->unitLambda=this->getParamsPtr()->unitLambda;
	paramsOut->units=this->getParamsPtr()->units;

	paramsOut->MTransform[0]=this->getParamsPtr()->MTransform.m11;
	paramsOut->MTransform[1]=this->getParamsPtr()->MTransform.m12;
	paramsOut->MTransform[2]=this->getParamsPtr()->MTransform.m13;
	paramsOut->MTransform[3]=this->getParamsPtr()->MTransform.m14;

	paramsOut->MTransform[4]=this->getParamsPtr()->MTransform.m21;
	paramsOut->MTransform[5]=this->getParamsPtr()->MTransform.m22;
	paramsOut->MTransform[6]=this->getParamsPtr()->MTransform.m23;
	paramsOut->MTransform[7]=this->getParamsPtr()->MTransform.m24;

	paramsOut->MTransform[8]=this->getParamsPtr()->MTransform.m31;
	paramsOut->MTransform[9]=this->getParamsPtr()->MTransform.m32;
	paramsOut->MTransform[10]=this->getParamsPtr()->MTransform.m33;
	paramsOut->MTransform[11]=this->getParamsPtr()->MTransform.m34;

	paramsOut->MTransform[12]=this->getParamsPtr()->MTransform.m41;
	paramsOut->MTransform[13]=this->getParamsPtr()->MTransform.m42;
	paramsOut->MTransform[14]=this->getParamsPtr()->MTransform.m43;
	paramsOut->MTransform[15]=this->getParamsPtr()->MTransform.m44;

	paramsOut->nrPixels[0]=this->getParamsPtr()->nrPixels.x;
	paramsOut->nrPixels[1]=this->getParamsPtr()->nrPixels.y;
	paramsOut->nrPixels[2]=this->getParamsPtr()->nrPixels.z;

	paramsOut->scale[0]=this->getParamsPtr()->scale.x;
	paramsOut->scale[1]=this->getParamsPtr()->scale.y;
	paramsOut->scale[2]=this->getParamsPtr()->scale.z;

	return FIELD_NO_ERR;
};


/**
 * \detail getParamsPtr 
 *
 * \param[in] void
 * 
 * \return fieldParms*
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldParams* Field::getParamsPtr(void)
{
	std::cout << "error in Field.getParamsPtr(): not defined for the given field representation" << std::endl;
	return NULL;
}

/**
 * \detail processParseResults 
 *
 * sets the parameters of the detector according to the given parse results
 *
 * \param[in] DetectorParseParamStruct &parseResults_Det
 * 
 * \return detError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::processParseResults(FieldParseParamStruct &parseResults_Src)
{
	std::cout << "error in Field.processParseResults(): not defined for the given Field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail detect 
 *
 * \param[in] Field **imagePtrPtr
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::convert2RayData(Field **imagePtrPtr, detParams &oDetParams)
{
	std::cout <<"error in Field.convert2RayData(): this has not yet been implemented for the given Field representation" << std::endl;
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
fieldError Field::traceScene(Group &oGroup, bool RunOnCPU)
{
	std::cout <<"error in Field.traceScene(): this has not yet been implemented for the given Field representation" << std::endl;
	return FIELD_ERR;
}

/**
 * \detail traceStep 
 *
 * \param[in] Group &oGroup
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::traceStep(Group &oGroup, bool RunOnCPU)
{
	std::cout <<"error in Field.traceStep(): this has not yet been implemented for the given Field representation" << std::endl;
	return FIELD_ERR;
}

/**
 * \detail createLayoutInstance
 *
 * \param[in] 
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError  Field::createLayoutInstance()
{
	std::cout <<"error in Field.createLayoutInstance(): this has not yet been implemented for the given Field representation" << std::endl;
	return FIELD_ERR;
};

void Field::setSimMode(SimMode &simMode)
{
	std::cout <<"error in Field.setSimMode(): this has not yet been implemented for the given Field representation" << std::endl;
};

/**
 * \detail doSim
 *
 * \param[in] Group &oGroup, simAssParams &params, bool &simDone
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError  Field::doSim(Group &oGroup, simAssParams &params, bool &simDone)
{
	std::cout <<"error in Field.doSim(): this has not yet been implemented for the given Field representation" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail initSimulation
 *
 * \param[in] Group &oGroup, simAssParams &params
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::initSimulation(Group &oGroup, simAssParams &params)
{
	std::cout <<"error in Field.initSimulation(): this has not yet been implemented for the given Field representation" << std::endl;
	return FIELD_ERR;
}

/**
 * \detail initLayout
 *
 * \param[in] Group &oGroup, simAssParams &params
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError Field::initLayout(Group &oGroup, simAssParams &params)
{
	std::cout <<"error in Field.initLayout(): this has not yet been implemented for the given Field representation" << std::endl;
	return FIELD_ERR;
}

void Field::setProgressCallback(void* p2CallbackObjectIn, void (*callbackProgressIn)(void* p2Object, int progressValue))
{
	this->p2ProgCallbackObject=p2CallbackObjectIn;
	this->callbackProgress=callbackProgressIn;
}

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
fieldError  Field::parseXml(pugi::xml_node &field, vector<Field*> &fieldVec, SimParams simParams)
{
	Parser_XML l_parser;
	if (!this->checkParserError(l_parser.attrByNameToDouble(field, "lambda", this->getParamsPtr()->lambda)))
		return FIELD_ERR;

	this->getParamsPtr()->lambda=this->getParamsPtr()->lambda*1e-6; // in our trace we use lambda in mm. in the gui we give lambda in nm...

	this->getParamsPtr()->pseudoBandwidth=0;
	this->getParamsPtr()->nrPseudoLambdas=1;
	return FIELD_NO_ERR;
};

/**
 * \detail checks wether parseing was succesfull and assembles the error message if it was not
 *
 * returns the coordinates of the minimum corner of the bounding box of the surface
 *
 * \param[in] char *msg
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool Field::checkParserError(char *msg)
{
	if (msg==NULL)
		return true;
	else
	{
		cout << "error in Field.parseXML(): " << msg << endl;
		delete msg;
		msg=NULL;
		return false;
	}
};