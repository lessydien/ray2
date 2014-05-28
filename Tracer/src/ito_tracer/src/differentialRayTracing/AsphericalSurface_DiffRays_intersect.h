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

/**\file AsphericalSurface_intersect.h
* \brief header file containing definitions of functions and datastructures that can be used on the GPU as well
* 
*           
* \author Mauch
*/

#ifndef ASPHERICALSURFACE_DIFFRAYS_INTERSECT_H
  #define ASPHERICALSURFACE_DIFFRAYS_INTERSECT_H
  
/* include header of basis class */
#include "Material_DiffRays_hit.h"
#include "../AsphericalSurface_intersect.h"
#include "../rayTracingMath.h"

/* declare class */
/**
  *\class   AsphericalSurface_DiffRays_ReducedParams 
  *\ingroup Geometry
  *\brief   reduced set of params that is calculated before the actual tracing from the full set of params. This parameter set will be loaded onto the GPU if the tracing is done there
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
class AsphericalSurface_DiffRays_ReducedParams : public AsphericalSurface_ReducedParams
{
  public:
	double3 tilt;
};

/**
 * \detail calcHitParamsAsphere_DiffRays 
 *
 * calculates the parameters of the surface at the intersection point that are needed within the hit-function
 *
 * \param[in] double3 position,AsphericalSurface_DiffRays_ReducedParams params
 * 
 * \return Mat_DiffRays_hitParams
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE Mat_DiffRays_hitParams calcHitParamsAsphere_DiffRays(double3 position,AsphericalSurface_DiffRays_ReducedParams params)
{
	Mat_DiffRays_hitParams t_hitParams_DiffRays;
	Mat_hitParams t_hitParams;
	t_hitParams=calcHitParamsAsphere(position, params);
	t_hitParams_DiffRays.normal=t_hitParams.normal;

    // calc main directions
    double x=position.x;
    double y=position.y;
    double z=position.z;
    double c2=params.c2;
    double c=params.c;
    double k=params.k;
    double c4=params.c4;
    double c6=params.c6;
    double c8=params.c8;
    double c10=params.c10;
    double c12=params.c12;
    double c14=params.c14;
    double c16=params.c16;


    // derivatives are calculated via the matlab symbolic toolbox. Therefore they look bulky but have a high chance to be correct...
    double dfdx=2*c2*x + 4*c4*y*(x*x + y*y) + 6*c6*y*pow(x*x + y*y,2) + 8*c8*x*pow(x*x + y*y,3) + 10*c10*x*pow(x*x + y*y,4) + 12*c12*x*pow(x*x + y*y,5) + 14*c14*x*pow(x*x + y*y,6) + 16*c16*x*pow(x*x + y*y,7) + (2*c*x)/(sqrt(- (x*x + y*y)*(k + 1)*c*c + 1) + 1) + (pow(c,3)*x*(x*x + y*y)*(k + 1))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,2)*sqrt(1 - c*c*(x*x + y*y)*(k + 1)));
    double dfdy=2*c2*y + 4*c4*y*(x*x + y*y) + 6*c6*y*pow(x*x + y*y,2) + 8*c8*y*pow(x*x + y*y,3) + 10*c10*y*pow(x*x + y*y,4) + 12*c12*y*pow(x*x + y*y,5) + 14*c14*y*pow(x*x + y*y,6) + 16*c16*y*pow(x*x + y*y,7) + (2*c*y)/(sqrt(- (x*x + y*y)*(k + 1)*c*c + 1) + 1) + (pow(c,3)*y*(x*x + y*y)*(k + 1))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,2)*sqrt(1 - c*c*(x*x + y*y)*(k + 1)));
    double dfdz=-1;

    double normFacNormal=(dfdx*dfdx+dfdy*dfdy+dfdz*dfdz);

    double normX=dfdx/sqrt(normFacNormal);
    double normY=dfdy/sqrt(normFacNormal);
    double normZ=dfdz/sqrt(normFacNormal);

    double normFac=(dfdx*dfdx+dfdy*dfdy);
    double tangZ=sqrt(normX*normX+normY*normY)*sqrt(normFac);

    t_hitParams_DiffRays.mainDirX=make_double3(normX, normY, tangZ);
    t_hitParams_DiffRays.mainDirX=normalize(t_hitParams_DiffRays.mainDirX);
    t_hitParams_DiffRays.mainDirY=cross(t_hitParams_DiffRays.normal, t_hitParams_DiffRays.mainDirX);
    t_hitParams_DiffRays.mainDirY=normalize(t_hitParams_DiffRays.mainDirY);

    // calc derivative of normal. Again we used the symbolic toolbox of matlab...
    double ddfddx = 2*c2 + 6*c6*pow(x*x + y*y,2) + 8*c8*pow(x*x + y*y,3) + 10*c10*pow(x*x + y*y,4) + 12*c12*pow(x*x + y*y,5) + 14*c14*pow(x*x + y*y,6) + 16*c16*pow(x*x + y*y,7) + (2*c)/(sqrt(- (x*x + y*y)*(k + 1)*c*c + 1) + 1) + 8*c4*x*x + 4*c4*(x*x + y*y) + 48*c8*x*x*pow(x*x + y*y,2) + 80*c10*x*x*pow(x*x + y*y,3) + 120*c12*x*x*pow(x*x + y*y,4) + 168*c14*x*x*pow(x*x + y*y,5) + 224*c16*x*x*pow(x*x + y*y,6) + 24*c6*x*x*(x*x + y*y) + (pow(c,3)*(x*x + y*y)*(k + 1))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,2)*sqrt(1 - c*c*(x*x + y*y)*(k + 1))) + (4*pow(c,3)*x*x*(k + 1))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,2)*sqrt(1 - c*c*(x*x + y*y)*(k + 1))) - (2*pow(c,5)*x*x*(x*x + y*y)*pow(k + 1,2))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,3)*(c*c*(x*x + y*y)*(k + 1) - 1)) + (pow(c,5)*x*x*(x*x + y*y)*pow(k + 1,2))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,2)*pow(1 - c*c*(x*x + y*y)*(k + 1),3/2));
    double ddfdxdy = 8*c4*x*y + 48*c8*x*y*pow(x*x + y*y,2) + 80*c10*x*y*pow(x*x + y*y,3) + 120*c12*x*y*pow(x*x + y*y,4) + 168*c14*x*y*pow(x*x + y*y,5) + 224*c16*x*y*pow(x*x + y*y,6) + 24*c6*x*y*(x*x + y*y) + (4*pow(c,3)*x*y*(k + 1))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,2)*sqrt(1 - c*c*(x*x + y*y)*(k + 1))) - (2*pow(c,5)*x*y*(x*x + y*y)*pow(k + 1,2))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,3)*(c*c*(x*x + y*y)*(k + 1) - 1)) + (pow(c,5)*x*y*(x*x + y*y)*pow(k + 1,2))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,2)*pow(1 - c*c*(x*x + y*y)*(k + 1),3/2));
    double ddfdxdz = 0;

    double ddfdydx = ddfdxdy; // for rotationally symmetric aspheres (even aspheres...)
    double ddfddy = 2*c2 + 6*c6*pow(x*x + y*y,2) + 8*c8*pow(x*x + y*y,3) + 10*c10*pow(x*x + y*y,4) + 12*c12*pow(x*x + y*y,5) + 14*c14*pow(x*x + y*y,6) + 16*c16*pow(x*x + y*y,7) + (2*c)/(sqrt(- (x*x + y*y)*(k + 1)*c*c + 1) + 1) + 8*c4*y*y + 4*c4*(x*x + y*y) + 48*c8*y*y*pow(x*x + y*y,2) + 80*c10*y*y*pow(x*x + y*y,3) + 120*c12*y*y*pow(x*x + y*y,4) + 168*c14*y*y*pow(x*x + y*y,5) + 224*c16*y*y*pow(x*x + y*y,6) + 24*c6*y*y*(x*x + y*y) + (pow(c,3)*(x*x + y*y)*(k + 1))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,2)*sqrt(1 - c*c*(x*x + y*y)*(k + 1))) + (4*pow(c,3)*y*y*(k + 1))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,2)*sqrt(1 - c*c*(x*x + y*y)*(k + 1))) - (2*pow(c,5)*y*y*(x*x + y*y)*pow(k + 1,2))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,3)*(c*c*(x*x + y*y)*(k + 1) - 1)) + (pow(c,5)*y*y*(x*x + y*y)*pow(k + 1,2))/(pow(sqrt(1 - c*c*(x*x + y*y)*(k + 1)) + 1,2)*pow(1 - c*c*(x*x + y*y)*(k + 1),3/2));
    double ddfdydz = 0;

    double ddfdzdx = 0;
    double ddfdzdy = 0;
    double ddfddz = 0;

    double3 dndx=make_double3(ddfddx/sqrt(normFacNormal)+dfdx*(-0.5*pow(normFacNormal,3/2))*2*(dfdx*ddfddx+dfdy*ddfdydx+dfdz*ddfdzdx),
                              ddfdydx/sqrt(normFacNormal)+dfdy*(-0.5*pow(normFacNormal,3/2))*2*(dfdx*ddfddx+dfdy*ddfdydx+dfdz*ddfdzdx),
                              ddfdzdx/sqrt(normFacNormal)+dfdz*(-0.5*pow(normFacNormal,3/2))*2*(dfdx*ddfddx+dfdy*ddfdydx+dfdz*ddfdzdx));
    double3 dndy=make_double3(ddfdxdy/sqrt(normFacNormal)+dfdx*(-0.5*pow(normFacNormal,3/2))*2*(dfdx*ddfdxdy+dfdy*ddfddy+dfdz*ddfdzdy),
                              ddfddy/sqrt(normFacNormal)+dfdz*(-0.5*pow(normFacNormal,3/2))*2*(dfdx*ddfdxdy+dfdy*ddfddy+dfdz*ddfdzdy),
                              ddfdzdy/sqrt(normFacNormal)+dfdz*(-0.5*pow(normFacNormal,3/2))*2*(dfdx*ddfdxdy+dfdy*ddfddy+dfdz*ddfdzdy));
    double3 dndz=make_double3(ddfdxdz/sqrt(normFacNormal)+dfdx*(-0.5*pow(normFacNormal,3/2))*2*(dfdx*ddfdxdz+dfdy*ddfdydz+dfdz*ddfddz),
                              ddfdydz/sqrt(normFacNormal)+dfdy*(-0.5*pow(normFacNormal,3/2))*2*(dfdx*ddfdxdz+dfdy*ddfdydz+dfdz*ddfddz),
                              ddfddz/sqrt(normFacNormal)+dfdz*(-0.5*pow(normFacNormal,3/2))*2*(dfdy*ddfdxdz+dfdy*ddfdydz+dfdz*ddfddz));



    t_hitParams_DiffRays.mainRad.x=1/sqrt(pow(dot(t_hitParams_DiffRays.mainDirX, dndx),2) + pow(dot(t_hitParams_DiffRays.mainDirX, dndy),2) + pow(dot(t_hitParams_DiffRays.mainDirX, dndz),2));
    t_hitParams_DiffRays.mainRad.y=1/sqrt(pow(dot(t_hitParams_DiffRays.mainDirY, dndx),2) + pow(dot(t_hitParams_DiffRays.mainDirY, dndy),2) + pow(dot(t_hitParams_DiffRays.mainDirY, dndz),2));

	return t_hitParams_DiffRays;
}

/**
 * \detail intersectRayAsphere_DiffRays 
 *
 * calculates the intersection of a differential ray with an aspherical surface
 *
 * \param[in] double3 rayPosition, double3 rayDirection, AsphericalSurface_DiffRays_ReducedParams params
 * 
 * \return double t
 * \sa 
 * \remarks this function is defined inline so it can be used on GPU and CPU
 * \author Mauch
 */
inline RT_HOSTDEVICE double intersectRayAsphere_DiffRays(double3 rayPosition, double3 rayDirection, AsphericalSurface_DiffRays_ReducedParams params)
{
	return intersectRayAsphere(rayPosition, rayDirection, params);
}


#endif
