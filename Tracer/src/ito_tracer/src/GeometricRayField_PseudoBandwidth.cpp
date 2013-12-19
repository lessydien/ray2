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

/**\file GeometricRayField_PseudoBandwidth_PseudoBandwidth.cpp
* \brief Rayfield for geometric raytracing
* 
*           
* \author Mauch
*/
#include <omp.h>
#include "GeometricRayField_PseudoBandwidth.h"
#include "myUtil.h"
#include "sampleConfig.h"
#include <iostream>
#include <string.h>
#include "Geometry.h"
#include "math.h"
#include "randomGenerator.h"
#include "Converter.h"
#include "MatlabInterface.h"
#include "DetectorLib.h"
#include <ctime>

using namespace optix;

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
fieldError GeometricRayField_PseudoBandwidth::convert2Intensity(Field* imagePtr, detParams &oDetParams)
{
	clock_t start, end;
	double msecs_Tracing=0;
	double msecs_Processing=0;
	double msecs=0;

	// start timing
	start=clock();

	//long2 l_GPUSubsetDim=calcSubsetDim();
	// cast the image to an IntensityField
	IntensityField* l_IntensityImagePtr=dynamic_cast<IntensityField*>(imagePtr);
	if (l_IntensityImagePtr == NULL)
	{
		std::cout << "error in GeometricRayField_PseudoBandwidth.convert2Intensity(): imagePtr is not of type IntensityField" << std::endl;
		return FIELD_ERR;
	}
		

	/************************************************************************************************************************** 
	* the idea in calculating the flux per pixel is as following:
	* first we create unit vectors along global coordinate axis. 
	* Then we scale this vectors with the scaling of the respective pixel.
	* Then we rotate and translate these vectors into the local coordinate system of the IntensityField
	* Finally we solve the equation system that expresses the ray position in terms of these rotated and scaled vectors.
	* floor() of the coefficients of these vectors gives the indices we were looking for                                      
	****************************************************************************************************************************/
	double3 scale=l_IntensityImagePtr->getParamsPtr()->scale;
	long3 nrPixels=l_IntensityImagePtr->getParamsPtr()->nrPixels;

	// create unit vectors
	double3 t_ez = make_double3(0,0,1);
	double3 t_ey=make_double3(0,1,0);
	double3 t_ex=make_double3(1,0,0);
	// transform unit vectors into local coordinate system of IntensityField
	rotateRay(&t_ez,oDetParams.tilt);
	rotateRay(&t_ey,oDetParams.tilt);
	rotateRay(&t_ex,oDetParams.tilt);

	// the origin of the IntensityField is at the outer edge of the detector rather than at the origin
	double3 offset;
	offset=oDetParams.root-oDetParams.apertureHalfWidth.x*t_ex;//+0.5*l_IntensityImagePtr->getParamsPtr()->scale*t_ex;
	offset=offset-oDetParams.apertureHalfWidth.y*t_ey;//+0.5*l_IntensityImagePtr->getParamsPtr()->scale*t_ey;
	offset=offset-0.005*t_ez;//+0.5*l_PhaseSpacePtr->getParamsPtr()->scale*t_ez;

	short solutionIndex;

	double3x3 Matrix=make_double3x3(t_ex,t_ey,t_ez);

	if (optix::det(Matrix)==0)
	{
		std::cout << "error in GeometricRayField_PseudoBandwidth.convert2Intensity(): Matrix is unitary!!" << std::endl;
		return FIELD_ERR; //matrix singular
	}
	double3x3 MatrixInv=inv(Matrix);

	unsigned long long hitNr=0;

//	double3 posMinOffset;
	double3 indexFloat;
	long3 index;
	if (this->rayParamsPtr->coherence==1) // sum coherently
	{
		complex<double> i_compl=complex<double>(0,1); // define complex number "i"

		for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
		{
			for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
			{
				unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
				// transform to local coordinate system
				double3 tmpPos=this->rayList[rayListIndex].position-offset;
				rotateRayInv(&tmpPos,oDetParams.tilt);

				index.x=floor((tmpPos.x)/scale.x);
				index.y=floor((tmpPos.y)/scale.y);
				index.z=floor((tmpPos.z)/0.02);


				// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
				if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) )  && (index.z==0) )
				{
					// use this ray only if it agrees with the ignoreDepth
					if ( this->rayList[rayListIndex].depth > oDetParams.ignoreDepth )
					{
						hitNr++;

						omp_set_num_threads(numCPU);

#pragma omp parallel default(shared) //shared(threadCounter)
{
		#pragma omp for schedule(dynamic, 10)
						for (signed long long jWvl=0; jWvl<this->rayParamsPtr->nrPseudoLambdas; jWvl++)
						{
							double wvl=(this->rayList[rayListIndex].lambda-this->rayParamsPtr->pseudoBandwidth/2+this->rayParamsPtr->pseudoBandwidth/this->rayParamsPtr->nrPseudoLambdas*jWvl);
							double phi=std::fmod(2*PI/wvl*this->rayList[rayListIndex].opl,2*M_PI);
							// we want to compute the field value at the centre of the pixel. Therefore we need to make som corrections in case the ray doesn't hit the Pixel at its centre
							// calc vector from differential ray to centre of pixel
							double3 PixelOffset=tmpPos-(index.x*t_ex+index.y*t_ey+index.z*t_ez);
							// calc projection of this vector onto the ray direction
							// calc ray direction on local coordinate system of detector
							double3 l_rayDir=this->rayList[rayListIndex].direction;
							rotateRayInv(&l_rayDir,oDetParams.tilt);
							double dz=dot(l_rayDir,PixelOffset);
							// calc additional phase at centre of pixel from linear approximation to local wavefront
							phi=phi+dz*2*M_PI/this->rayList[rayListIndex].lambda;

							complex<double> l_U=complex<double>(this->rayList[rayListIndex].flux*cos(phi),this->rayList[rayListIndex].flux*sin(phi));
							//l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+l_U; // create a complex amplitude from the rays flux and opl and sum them coherently
							l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+jWvl*nrPixels.x*nrPixels.y]=l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+jWvl*nrPixels.x*nrPixels.y]+l_U; // create a complex amplitude from the rays flux and opl and sum them coherently
						}
}
					}
				}
			}
		}

		//for (unsigned long long j=0;j<rayListLength;j++)
		//{
		//	posMinOffset=this->rayList[j].position-offset;
		//	indexFloat=MatrixInv*posMinOffset;
		//	index.x=floor(indexFloat.x+0.5);
		//	index.y=floor(indexFloat.y+0.5);
		//	index.z=floor(indexFloat.z+0.5);
		//	
		//	// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
		//	if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
		//	{
		//		complex<double> l_exp=complex<double>(0,2*PI/this->rayList[j].lambda*this->rayList[j].opl);
		//		l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=l_IntensityImagePtr->getComplexAmplPtr()[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+this->rayList[j].flux*c_exp(l_exp); // create a complex amplitude from the rays flux and opl and sum them coherently
		//	}
		//}
						omp_set_num_threads(numCPU);

#pragma omp parallel default(shared) //shared(threadCounter)
{
		#pragma omp for schedule(dynamic, 10)

		// loop through the pixels and calc intensity from complex amplitudes
		for (signed long long jx=0;jx<nrPixels.x;jx++)
		{
			for (signed long long jy=0;jy<nrPixels.y;jy++)
			{
				for (signed long long jWvl=0; jWvl<this->rayParamsPtr->nrPseudoLambdas; jWvl++)
				{
					// intensity is square of modulus of complex amplitude
					(l_IntensityImagePtr->getIntensityPtr())[jx+jy*nrPixels.x+jWvl*nrPixels.x*nrPixels.y]=pow(abs(l_IntensityImagePtr->getComplexAmplPtr()[jx+jy*nrPixels.x+jWvl*nrPixels.x*nrPixels.y]),2);
				}
			}
		}
}
	}
	else 
	{
		if (this->rayParamsPtr->coherence == 0)// sum incoherently
		{
			
			for (unsigned long long jy=0; jy<this->rayParamsPtr->GPUSubset_height; jy++)
			{
				for (unsigned long long jx=0; jx<this->rayParamsPtr->GPUSubset_width; jx++)
				{

					unsigned long long rayListIndex=jx+jy*GPU_SUBSET_WIDTH_MAX;
					// transform to local coordinate system
					double3 tmpPos=this->rayList[rayListIndex].position-offset;
					rotateRayInv(&tmpPos,oDetParams.tilt);

					rayStruct rayTest=this->rayList[rayListIndex];
					//posMinOffset=this->rayList[rayListIndex].position-offset;
					//indexFloat=MatrixInv*posMinOffset;
					// subtract half a pixel (0.5*scale.x). This way the centre of our pixels do not lie on the edge of the aperture but rather half a pixel inside...
					// then round to nearest neighbour
					//index.x=floor((indexFloat.x-0.5*scale.x)/scale.x+0.5);
					index.x=floor((tmpPos.x)/scale.x);
					index.y=floor((tmpPos.y)/scale.y);
					index.z=floor((tmpPos.z)/0.02);
					
					// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
					if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && (index.z==0) )
					{
						// use this ray only if it agrees with the ignoreDepth
						if ( this->rayList[rayListIndex].depth > oDetParams.ignoreDepth )
						{
							hitNr++;
							for (unsigned long long jWvl=0; jWvl<this->rayParamsPtr->nrPseudoLambdas; jWvl++)
							{
								(l_IntensityImagePtr->getIntensityPtr())[index.x+index.y*nrPixels.x+jWvl*nrPixels.x*nrPixels.y]+=this->rayList[rayListIndex].flux;								
							}
						}
					}
				}
			}


			//for (unsigned long long j=0;j<rayListLength;j++)
			//{
			//	posMinOffset=this->rayList[j].position-offset;
			//	indexFloat=MatrixInv*posMinOffset;
			//	index.x=floor(indexFloat.x+0.5);
			//	index.y=floor(indexFloat.y+0.5);
			//	index.z=floor(indexFloat.z+0.5);
			//	
			//	// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
			//	if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
			//	{
			//		hitNr++;
			//		(l_IntensityImagePtr->getIntensityPtr())[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+=this->rayList[j].flux;
			//	}
			//	//else
			//	//{
			//	//	std::cout <<  "ray number " << j << " did not hit target." << "x: " << rayList[j].position.x << ";y: " << rayList[j].position.y << "z: " << rayList[j].position.z << ";geometryID " << rayList[j].currentGeometryID << std::endl;
			//	//}
			//}
		}
		else
		{
			std::cout << "error in GeometricRayField_PseudoBandwidth.convert2Intensity(): partial coherence not implemented yet" << std::endl;
			return FIELD_ERR;
		}

	}
	// end timing
	end=clock();
	msecs=((end-start)/(double)CLOCKS_PER_SEC*1000.0);
	msecs_Tracing=msecs_Tracing+msecs;
	std::cout << " " << msecs <<"ms to process " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays." << std::endl;

	std::cout << " " << hitNr << " out of " << this->rayParamsPtr->GPUSubset_height*this->rayParamsPtr->GPUSubset_width << " rays in target" << std::endl;

	return FIELD_NO_ERR;
};

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
fieldError GeometricRayField_PseudoBandwidth::convert2ScalarField(Field* imagePtr, detParams &oDetParams)
{
	std::cout << "error in GeometricRayField_PseudoBandwidth.convert2ScalarField(): conversion to scalar field not yet implemented" << std::endl;
	return FIELD_ERR;
};

/**
 * \detail convert2VecField 

 *
 * \param[in] Field* imagePtr, detParams &oDetParams
 * 
 * \return fieldError
 * \sa 
 * \remarks 
 * \author Mauch
 */
fieldError GeometricRayField_PseudoBandwidth::convert2VecField(Field* imagePtr, detParams &oDetParams)
{
	std::cout << "error in GeometricRayField_PseudoBandwidth.convert2VecField(): conversion to vectorial field not yet implemented" << std::endl;
	return FIELD_ERR;
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
fieldError  GeometricRayField_PseudoBandwidth::parseXml(pugi::xml_node &field, vector<Field*> &fieldVec)
{
	Parser_XML l_parser;
	// call base class function
	if (FIELD_NO_ERR != GeometricRayField::parseXml(field, fieldVec))
	{
		std::cout << "error in GeometricRayField_PseudoBandwidth.parseXml(): RayField.parseXml()  returned an error." << std::endl;
		return FIELD_ERR;
	}

	if ( (this->rayParamsPtr->dirDistrType == RAYDIR_GRID_RECT) || (this->rayParamsPtr->dirDistrType == RAYDIR_GRID_RAD) )
	{
		std::cout << "error in GeometricRayField_PseudoBandwidth.parseXml(): RAYDIR_GRID_RAD and RAYDIR_GRID_RECT are not allowed for geometric ray fields" << std::endl;
		return FIELD_ERR;
	}

	if ((l_parser.attrByNameToDouble(field, "pseudoBandwidth", this->getParamsPtr()->pseudoBandwidth)))
		this->getParamsPtr()->pseudoBandwidth=0; // default to zero
	// bandwidth has to be transformed from nm to mm
	this->getParamsPtr()->pseudoBandwidth=this->getParamsPtr()->pseudoBandwidth*1e-6;
	if ((l_parser.attrByNameToInt(field, "nrPseudoLambdas", this->getParamsPtr()->nrPseudoLambdas)))
		this->getParamsPtr()->nrPseudoLambdas=1; // default to one

	this->rayParamsPtr->totalLaunch_height=this->rayParamsPtr->height;
	this->rayParamsPtr->totalLaunch_width=this->rayParamsPtr->width;
	this->rayParamsPtr->nrRayDirections=make_ulong2(1,1);

	return FIELD_NO_ERR;
};