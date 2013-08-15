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

/**\file ConverterMath.cpp
* \brief collection of functions to convert one field representation to another
* 
*           
* \author Mauch
*/

#include "converterMath.h"
#include "math.h"
#include <iostream>

using namespace optix;
/**
 * \detail geomRays2IntensityCPU
 *
 * converts geometric ray representation of light field into Intensity representation
 * this function became obsolete as we were moving the converter functions into the field representations themselves...
 *
 * \param[in] rayStruct* rayListPtr, unsigned long long rayListLength, double* IntensityPtr, double4x4 MTransform, double3 scale, long3 nrPixels
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool geomRays2IntensityCPU(rayStruct* rayListPtr, unsigned long long rayListLength, double* IntensityPtr, double4x4 MTransform, double3 scale, long3 nrPixels, double coherence)
{
	/************************************************************************************************************************** 
	* the idea in calculating the flux per pixel is as following:
	* first we create unit vectors along global coordinate axis. 
	* Then we scale this vectors with the scaling of the respective pixel.
	* Then we rotate and translate these vectors into the local coordinate system of the IntensityField
	* Finally we solve the equation system that expresses the ray position in terms of these rotated and scaled vectors.
	* floor() of the coefficients of these vectors gives the indices we were looking for                                      
	****************************************************************************************************************************/
	double3 test3=scale;
	double4x4 test4x4=MTransform;
	// save the offset from the transformation matrix
	double3 offset=make_double3(MTransform.m14, MTransform.m24, MTransform.m34);
	// set offset in transformation matrix to zero for rotation of the scaled unit vectors
	MTransform.m14=0;
	MTransform.m24=0;
	MTransform.m34=0;
	// scale unit vectors
	double3 t_ez = make_double3(0,0,1)*scale.z; // create vector pointing in z-direction 
	double3 t_ey=make_double3(0,1,0)*scale.y; // create vector pointing in y-direction
	double3 t_ex=make_double3(1,0,0)*scale.x; // create vector pointing in x-direction
	// transform unit vectors into local coordinate system of IntensityField
	t_ez=MTransform*t_ez;
	t_ey=MTransform*t_ey;
	t_ex=MTransform*t_ex;

	short solutionIndex;

	double3x3 Matrix=make_double3x3(t_ex,t_ey,t_ez);
	if (optix::det(Matrix)==0)
	{
		std::cout << "error in ConverterMath.geomRays2Intensity(): Matrix is unitary!!" << std::endl;
		return false; //matrix singular
	}
	double3x3 MatrixInv=inv(Matrix);
	double3 posMinOffset;
	double3 indexFloat;
	long3 index;
	unsigned long long j=0;
	if (coherence==1) // sum coherently
	{
		complex<double> *ComplAmplPtr = (complex<double>*)calloc(nrPixels.x*nrPixels.y*nrPixels.z, sizeof(complex<double>));
		complex<double> i_compl=complex<double>(0,1); // define complex number "i"
		for (j=0;j<rayListLength;j++)
		{
			posMinOffset=rayListPtr[j].position-offset;
			indexFloat=MatrixInv*posMinOffset;
			index.x=floor(indexFloat.x+0.5);
			index.y=floor(indexFloat.y+0.5);
			index.z=floor(indexFloat.z+0.5);
			
			// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
			if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
			{
				complex<double> l_exp=polar(0.0,2*PI/rayListPtr[j].lambda*rayListPtr[j].opl);
				ComplAmplPtr[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]=ComplAmplPtr[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+rayListPtr[j].flux*l_exp; // create a complex amplitude from the rays flux and opl and sum them coherently
			}
		}
		// loop through the pixels and calc intensity from complex amplitudes
		for (unsigned long long jx=0;jx<nrPixels.x;jx++)
		{
			for (unsigned long long jy=0;jy<nrPixels.y;jy++)
			{
				for (unsigned long long jz=0;jz<nrPixels.z;jz++)
				{
					// intensity is square of modulus of complex amplitude
					IntensityPtr[jx+jy*nrPixels.x+jz*nrPixels.x*nrPixels.y]=pow(abs(ComplAmplPtr[jx+jy*nrPixels.x+jz*nrPixels.x*nrPixels.y]),2);
				}
			}
		}

	}
	else 
	{
		if (coherence == 0)// sum incoherently
		{
			for (j=0;j<rayListLength;j++)
			{
				posMinOffset=rayListPtr[j].position-offset;
				indexFloat=MatrixInv*posMinOffset;
				index.x=floor(indexFloat.x+0.5);
				index.y=floor(indexFloat.y+0.5);
				index.z=floor(indexFloat.z+0.5);
				
				// use this ray only if it is inside our Intensity Field. Otherwise ignore it...
				if ( ( (index.x<nrPixels.x)&&(index.x>=0) ) && ( (index.y<nrPixels.y)&&(index.y>=0) ) && ( (index.z<nrPixels.z)&&(index.z>=0) ) )
					IntensityPtr[index.x+index.y*nrPixels.x+index.z*nrPixels.x*nrPixels.y]+=rayListPtr[j].flux;
			}
		}
		else
		{
			std::cout << "error in geomRays2Intensity(): partial coherence not implemented yet" << std::endl;
			return false;
		}
	}
	return true;
};

/**
 * \detail gaussBeams2ScalarFieldCPU
 *
 * converts gaussian beam ray representation of light field into scalar field representation
 *
 * \param[in] rayStruct* rayListPtr, unsigned long long rayListLength, double* IntensityPtr, double4x4 MTransform, double3 scale, long3 nrPixels
 * 
 * \return bool
 * \sa 
 * \remarks 
 * \author Mauch
 */
bool gaussBeams2ScalarFieldCPU(gaussBeamRayStruct *rayList, unsigned long long rayListLength, complex<double>* U, fieldParams* params)
{
	unsigned int i=0; // loop counter
	complex<double> i_compl=complex<double>(0,1); // define complex number "i"
	// loop through rayList
	for (i=0;i<rayListLength;i++)
	{
		// reference parabasal rays to base ray
		double waistRayXx=(rayList[i].waistRayX.position.x-rayList[i].baseRay.position.x);
		//double waistRayXy=(rayList[i].waistRayX.position.y-rayList[i].baseRay.position.y);
		//double waistRayXz=(rayList[i].waistRayX.position.z-rayList[i].baseRay.position.z);
		//double waistRayYx=(rayList[i].waistRayY.position.x-rayList[i].baseRay.position.x);
		double waistRayYy=(rayList[i].waistRayY.position.y-rayList[i].baseRay.position.y);
		//double waistRayYz=(rayList[i].waistRayY.position.z-rayList[i].baseRay.position.z);
		//double waistRayXdivTanY=tan(acos(rayList[i].waistRayX.direction.y)-acos(rayList[i].baseRay.direction.y));
		double waistRayXdivTanX=tan(acos(rayList[i].waistRayX.direction.x)-acos(rayList[i].baseRay.direction.x));
		//double waistRayXdivTanZ=tan(acos(rayList[i].waistRayX.direction.z)-acos(rayList[i].baseRay.direction.z));
		double waistRayYdivTanY=tan(acos(rayList[i].waistRayY.direction.y)-acos(rayList[i].baseRay.direction.y));
		//double waistRayYdivTanX=tan(acos(rayList[i].waistRayY.direction.x)-acos(rayList[i].baseRay.direction.x));
		//double waistRayYdivTanZ=tan(acos(rayList[i].waistRayY.direction.z)-acos(rayList[i].baseRay.direction.z));
		double divRayXx=(rayList[i].divRayX.position.x-rayList[i].baseRay.position.x);
		//double divRayXy=(rayList[i].divRayX.position.y-rayList[i].baseRay.position.y);
		//double divRayXz=(rayList[i].divRayX.position.z-rayList[i].baseRay.position.z);
		//double divRayYx=(rayList[i].divRayY.position.x-rayList[i].baseRay.position.x);
		double divRayYy=(rayList[i].divRayY.position.y-rayList[i].baseRay.position.y);
		//double divRayYz=(rayList[i].divRayY.position.z-rayList[i].baseRay.position.z);
		//double divXcosX=rayList[i].divRayX.direction.x;
		//double divXcosY=rayList[i].divRayX.direction.y;
		//double divXcosZ=rayList[i].divRayX.direction.z;
		//double divYcosX=rayList[i].divRayY.direction.x;
		//double divYcosY=rayList[i].divRayY.direction.y;
		//double divYcosZ=rayList[i].divRayY.direction.z;
		//double baseDivX=rayList[i].baseRay.direction.x;
		//double baseDivY=rayList[i].baseRay.direction.y;
		//double baseDivZ=rayList[i].baseRay.direction.z;
		double divRayXdivTanX=tan(acos(rayList[i].divRayX.direction.x)-acos(rayList[i].baseRay.direction.x));
		//double divRayXdivTanY=tan(acos(rayList[i].divRayX.direction.y)-acos(rayList[i].baseRay.direction.y));
		//double divRayXdivTanZ=tan(acos(rayList[i].divRayX.direction.z)-acos(rayList[i].baseRay.direction.z));
		//double divRayYdivTanX=tan(acos(rayList[i].divRayY.direction.x)-acos(rayList[i].baseRay.direction.x));
		double divRayYdivTanY=tan(acos(rayList[i].divRayY.direction.y)-acos(rayList[i].baseRay.direction.y));
		//double divRayYdivTanZ=tan(acos(rayList[i].divRayY.direction.z)-acos(rayList[i].baseRay.direction.z));
		
		// calculate physical beam parameters from ray data: /* see "R.Herloski, Gaussian beam ray-equivalent modeling and optical design, Applied Optics 1983, Vol 22, pp. 1168-1174" for reference */
	    // calculate distance to beam waist
		double z0gX=(divRayXx*divRayXdivTanX+waistRayXx*waistRayXdivTanX)/(pow(divRayXdivTanX,2)+pow(waistRayXdivTanX,2));
		double z0gY=(divRayYy*divRayYdivTanY+waistRayYy*waistRayYdivTanY)/(pow(divRayYdivTanY,2)+pow(waistRayYdivTanY,2));

		// calculate far field divergence
		double divTanX=sqrt(pow(divRayXdivTanX,2)+pow(waistRayXdivTanX,2));
		double divTanY=sqrt(pow(divRayYdivTanY,2)+pow(waistRayYdivTanY,2));
		// calculate beam waists
		//double wgX=sqrt(pow(divRayXx,2)+pow(waistRayXx,2));
		//double wgY=sqrt(pow(divRayYy,2)+pow(waistRayYy,2));
		double w0gX=rayList[i].baseRay.lambda/(PI*divTanX);
		double w0gY=rayList[i].baseRay.lambda/(PI*divTanY);
		//double zX=wgX*z0gX/w0gX;
		//double zY=wgY*z0gY/w0gY;
		complex<double> qgX, qgY;
		qgX=complex<double>(z0gX,PI*pow(w0gX,2)/rayList[i].baseRay.lambda);
		//qgX.real=z0gX;
		//qgX.imag=PI*pow(w0gX,2)/rayList[i].baseRay.lambda;
		qgY=complex<double>(z0gY,PI*pow(w0gY,2)/rayList[i].baseRay.lambda);
		//qgY.real=z0gY;
		//qgY.imag=PI*pow(w0gY,2)/rayList[i].baseRay.lambda;

		double3 ek1=make_double3(rayList[i].baseRay.direction.x, rayList[i].baseRay.direction.y, rayList[i].baseRay.direction.z);
		// find the beam waist positions along the beam trajectory
		double3 rw01x=rayList[i].baseRay.position-qgX.real()*ek1;
		double3 rw01y=rayList[i].baseRay.position-qgY.real()*ek1;
		// loop through the points of the observation plane
		unsigned int ix=0;
		unsigned int iy=0;
		for (ix=0;ix<params->nrPixels.x;ix++)
		{
			for (iy=0;iy<params->nrPixels.y;iy++)
			{
				// define the vector to the screen points
				double3 r=make_double3(params->MTransform.m14+ix*params->scale.x, params->MTransform.m24+iy*params->scale.y, rayList[i].baseRay.position.z);
				//double absr=length(r);
	            
				// calc complex beam parameter of gaussian beam corresponding to current point
				complex<double> q1x, q1y;
				q1x=complex<double>(dot(ek1,(r-rw01x)),PI*pow(w0gX,2)/rayList[i].baseRay.lambda);
				//q1x.real= dot(ek1,(r-rw01x));
				//q1x.imag= PI*pow(w0gX,2)/rayList[i].baseRay.lambda;
				q1y=complex<double>(dot(ek1,(r-rw01y)),PI*pow(w0gY,2)/rayList[i].baseRay.lambda);
				//q1y.real = dot(ek1,(r-rw01y));
				//q1y.imag=PI*pow(w0gY,2)/rayList[i].baseRay.lambda;

				// calc vector from point r to beam waist position rw0x
				double3 rrToW=r-rw01x;
				// calc distance of point r to beam waist position rw0x
				//double distrToRw01x=length(r-rw01x);
				//double distrToRw01y=length(r-rw01y);
				// calc vector to beam centre
				double3 rCentre=dot(rrToW,ek1)*ek1+rw01x;
				// calc transverse vector
				double3 rt=rCentre-r;
	            
	//%             % calc transverse distance of point r to beam axis
	//%             rt1x=sqrt(distrToRw01x^2-(ek1*(r-rw01x))^2);
	//%             rt1y=sqrt(distrToRw01y^2-(ek1*(r-rw01y))^2);

				// calc the transverse distance along transverse-x and transverse-y
				double etz=sqrt(pow(ek1.x,2)/(pow(ek1.x,2)+pow(ek1.z,2)));
				double etx=sqrt(1-pow(etz,2));
				double distX=dot(make_double3(etx, 0, etz),rt);
				double distY=dot(make_double3(0, etx, etz),rt);
				// calc field at point in gaussian beam from physical beam parameters and geometrical OPD
				double k = 2*PI/rayList[i].baseRay.lambda;
				//complex<double> exp1=-i_compl*(k*pow(distX,2))/(2*q1x);
				complex<double> exp1=(k*pow(distX,2))/(2.0*q1x);
				//complex<double> exp2=-i_compl*(k*q1x.real());
				double exp2=-(k*q1x.real());
				//complex<double> exp3=-i_compl*(k*(rayList[i].baseRay.opl));
				double exp3=-(k*(rayList[i].baseRay.opl));
				//complex<double> exp4=-i_compl*(k*pow(distY,2))/(2*q1y);
				complex<double> exp4=-(k*pow(distY,2))/(2.0*q1y);
				//complex<double> exp5=-i_compl*(k*q1y.real());
				double exp5=-(k*q1y.real());
				//complex<double> exp6=-i_compl*(k*(rayList[i].baseRay.opl));
				double exp6=-(k*(rayList[i].baseRay.opl));
				//complex<double> test1=c_mul(c_sqrt(q1x),c_sqrt(q1y));
				//complex<double> test2=c_div(complex<double>(1,0),test1);
				//complex<double> test3=c_add(exp1,exp2);
				//complex<double> test4=c_add(test3,exp3);
				//complex<double> test5=c_add(test4,exp4);
				//complex<double> test6=c_add(test5,exp5);
				//complex<double> test7=c_add(test6,exp6);
				//complex<double> test8=c_exp(test7);
				//complex<double> fieldValTest=c_mul(test2,test8);
				//complex<double> fieldVal=1/(sqrt(q1x)*sqrt(q1y))*polar(1,exp1+exp2+exp3+exp4+exp5+exp6);
				complex<double> fieldVal=1.0/(sqrt(q1x)*sqrt(q1y))*polar(1.0,exp2+exp3+exp5+exp6)*exp1*exp4;
				// FieldGeometricArr(jx,jy,jBeamlet)=1/sqrt(q1x)*exp(-i*rayList[i].baseRay.k*distX^2/(2*q1x))*exp(-i*rayList[i].baseRay.k*real(q1x))*exp(-i*rayList[i].baseRay.k*(rayList[i].baseRay.OPL)).*1/sqrt(q1y)*exp(-i*rayList[i].baseRay.k*distY^2/(2*q1y))*exp(-i*rayList[i].baseRay.k*real(q1y))*exp(-i*rayList[i].baseRay.k*(rayList[i].baseRay.OPL));
				//U[ix+iy*params->sizeX]=U[ix+iy*params->sizeX]+fieldVal;//exp4;//complex<double>(distY,0);//
				U[ix+iy*params->nrPixels.x]=U[ix+iy*params->nrPixels.x]+(fieldVal);
				//rayList[i].baseRay.flux
			}
		} // end loop through points of observation plane

	}// end loop through rayList
	return true; //indicate no error
/* matlab code
function result=calcFieldFromGeometricBeamlets(GBgeometric,X,Y);
nrBeamlets=length(GBgeometric);
sizeX=size(X);
%global FieldGeometricArr;
FieldGeometricArr=zeros(sizeX(2),sizeX(1),nrBeamlets);
for jBeamlet=1:1:nrBeamlets
    params.lambda=2*PI/rayList[i].baseRay.k;
    % reference parabasal rays to base ray
    waistRayXx=(rayList[i].waistRayX.position.x-rayList[i].baseRay.position.x);
    waistRayXy=(rayList[i].waistRayX.xyz(2)-rayList[i].baseRay.position.y);
    waistRayXz=(rayList[i].waistRayX.xyz(3)-rayList[i].baseRay.position.z);
    waistRayYx=(rayList[i].waistRayY.position.x-rayList[i].baseRay.position.x);
    waistRayYy=(rayList[i].waistRayY.position.y-rayList[i].baseRay.position.y);
    waistRayYz=(rayList[i].waistRayY.position.z-rayList[i].baseRay.position.z);
    waistRayXdivTanY=tan(acos(rayList[i].waistRayX.direction.y)-acos(rayList[i].baseRay.direction.y));
    waistRayXdivTanX=tan(acos(rayList[i].waistRayX.direction.x)-acos(rayList[i].baseRay.direction.x));
    waistRayXdivTanZ=tan(acos(rayList[i].waistRayX.direction.z)-acos(rayList[i].baseRay.direction.z));
    waistRayYdivTanY=tan(acos(rayList[i].waistRayY.direction.y)-acos(rayList[i].baseRay.direction.y));
    waistRayYdivTanX=tan(acos(rayList[i].waistRayY.direction.x)-acos(rayList[i].baseRay.direction.x));
    waistRayYdivTanZ=tan(acos(rayList[i].waistRayY.direction.z)-acos(rayList[i].baseRay.direction.z));
    divRayXx=(rayList[i].divRayX.position.x-rayList[i].baseRay.position.x);
    divRayXy=(rayList[i].divRayX.position.y-rayList[i].baseRay.position.y);
    divRayXz=(rayList[i].divRayX.position.z-rayList[i].baseRay.position.z);
    divRayYx=(rayList[i].divRayY.position.x-rayList[i].baseRay.position.x);
    divRayYy=(rayList[i].divRayY.position.y-rayList[i].baseRay.position.y);
    divRayYz=(rayList[i].divRayY.position.z-rayList[i].baseRay.position.z);
    divXcosX=rayList[i].divRayX.direction.x;
    divXcosY=rayList[i].divRayX.direction.y;
    divXcosZ=rayList[i].divRayX.direction.z;
    divYcosX=rayList[i].divRayY.direction.x;
    divYcosY=rayList[i].divRayY.direction.y;
    divYcosZ=rayList[i].divRayY.direction.z;
    baseDivX=rayList[i].baseRay.direction.x;
    baseDivY=rayList[i].baseRay.direction.y;
    baseDivZ=rayList[i].baseRay.direction.z;
    divRayXdivTanX=tan(acos(rayList[i].divRayX.direction.x)-acos(rayList[i].baseRay.direction.x));
    divRayXdivTanY=tan(acos(rayList[i].divRayX.direction.y)-acos(rayList[i].baseRay.direction.y));
    divRayXdivTanZ=tan(acos(rayList[i].divRayX.direction.z)-acos(rayList[i].baseRay.direction.z));
    divRayYdivTanX=tan(acos(rayList[i].divRayY.direction.x)-acos(rayList[i].baseRay.direction.x));
    divRayYdivTanY=tan(acos(rayList[i].divRayY.direction.y)-acos(rayList[i].baseRay.direction.y));
    divRayYdivTanZ=tan(acos(rayList[i].divRayY.direction.z)-acos(rayList[i].baseRay.direction.z));
    
    % transform geometric rays back to physical beam parameters
%     w0gX1=(waistRayXx...
%         *divRayXdivTanX...
%         -waistRayXdivTanX...
%         *divRayXx)...
%         /sqrt(divRayXdivTanX^2+...
%         waistRayXdivTanX^2);
%     w0gY1=(waistRayYy...
%         *divRayYdivTanY...
%         -waistRayYdivTanY...
%         *divRayYy)...
%         /sqrt(divRayYdivTanY^2+...
%         waistRayYdivTanY^2);
    % calculate distance to beam waist
    z0gX=(divRayXx...
        *divRayXdivTanX...
        +waistRayXx...
        *waistRayXdivTanX)...
        /(divRayXdivTanX^2+...
        waistRayXdivTanX^2);
    z0gY=(divRayYy...
        *divRayYdivTanY...
        +waistRayYy...
        *waistRayYdivTanY)...
        /(divRayYdivTanY^2+...
        waistRayYdivTanY^2);
    % calculate far field divergence
    divTanX=sqrt(divRayXdivTanX^2+waistRayXdivTanX^2);
    divTanY=sqrt(divRayYdivTanY^2+waistRayYdivTanY^2);
    wgX=sqrt(divRayXx^2+waistRayXx^2);
    wgY=sqrt(divRayYy^2+waistRayYy^2);
    w0gX=params.lambda/(PI*divTanX);
    w0gY=params.lambda/(PI*divTanY);
    zX=wgX*z0gX/w0gX;
    zY=wgY*z0gY/w0gY;
    qgX=z0gX+i*PI*w0gX^2/params.lambda;
    qgY=z0gY+i*PI*w0gY^2/params.lambda;

    % calc unit vector of beam propagation
%     FieldGeometricArr(jBeamlet,:)=1/sqrt(qg)*exp(-i*k*(x-rayList[i].baseRay.position.y).^2./(2*qg)); 
%     ek1=[sin(atan(rayList[i].baseRay.direction.z))*cos(atan(rayList[i].baseRay.direction.x));...
%         sin(atan(rayList[i].baseRay.direction.z))*sin(atan(rayList[i].baseRay.direction.x));...
%         cos(atan(rayList[i].baseRay.direction.z))];
    ek1=[rayList[i].baseRay.direction.x;...
        rayList[i].baseRay.direction.y;...
        rayList[i].baseRay.direction.z];
    % find the beam waist positions along the beam trajectory
    rw01x=[rayList[i].baseRay.position.x; rayList[i].baseRay.position.y; rayList[i].baseRay.position.z ]-real(qgX)*[ek1];
    rw01y=[rayList[i].baseRay.position.x; rayList[i].baseRay.position.y; rayList[i].baseRay.position.z ]-real(qgY)*[ek1];
    % loop through the points of the observation plane
    for jx=1:1:sizeX(1)
        for jy=1:1:sizeX(2)
            % define the vector to the screen points
            r=[X(jx,jy);Y(jx,jy);rayList[i].baseRay.position.z];
            absr=(sqrt(r(1,:)^2+r(2,:)^2+r(3,:)^2));
            
            % calc complex beam parameter of gaussian beam corresponding to
            % current point
            q1x = (ek1'*(r-rw01x))+i*PI*w0gX^2/params.lambda;
            q1y = (ek1'*(r-rw01y))+i*PI*w0gY^2/params.lambda;

            % calc vector from point r to beam waist position rw0x
            rrToW=r-rw01x;
            % calc distance of point r to beam waist position rw0x
            distrToRw01x=sqrt((r-rw01x)'*(r-rw01x));
            distrToRw01y=sqrt((r-rw01y)'*(r-rw01y));
            % calc vector to beam centre
            rCentre=(rrToW'*ek1)*ek1+rw01x;
            % calc transverse vector
            rt=rCentre-r;
            
%             % calc transverse distance of point r to beam axis
%             rt1x=sqrt(distrToRw01x^2-(ek1*(r-rw01x))^2);
%             rt1y=sqrt(distrToRw01y^2-(ek1*(r-rw01y))^2);

            % calc the transverse distance along transverse-x and transverse-y
            etz=sqrt(ek1(1)^2/(ek1(1)^2+ek1(3)^2));
            etx=sqrt(1-etz^2);
            distX=[etx 0 etz]*rt;
            distY=[0 etx etz]*rt;
            % calc field at point in gaussian beam from physical beam
            % parameters and geometrical OPD
            FieldGeometricArr(jx,jy,jBeamlet)=1/sqrt(q1x)*exp(-i*rayList[i].baseRay.k*distX^2/(2*q1x))*exp(-i*rayList[i].baseRay.k*real(q1x))*exp(-i*rayList[i].baseRay.k*(rayList[i].baseRay.OPL))...
                .*1/sqrt(q1y)*exp(-i*rayList[i].baseRay.k*distY^2/(2*q1y))*exp(-i*rayList[i].baseRay.k*real(q1y))*exp(-i*rayList[i].baseRay.k*(rayList[i].baseRay.OPL));
        end
    end
end
% do the summation of the beamlets
result=sum(FieldGeometricArr,3);
*/
};

