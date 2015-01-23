#ifndef SURFACEPARAMS_H
#define SURFACEPARAMS_H

#include <vector_types.h>

class Geometry_Params
{

};

class AsphericalSurface_Params : public Geometry_Params
{
  public:
  double3 vertex;
  double3 orientation;
  double k;
  double c;
  double c2;
  double c4;
  double c6;
  double c8;
  double c10;
};

class SphericalSurface_Params : public Geometry_Params
{
	public:
	  double3 centre;
	  double3 orientation;
	  double curvatureRadius;
	  double apertureRadius;
};

class PlaneSurface_Params : public Geometry_Params
{
  public:
   double3 root;
   double3 normal;
};

typedef struct 
{
  double3 vertex;
  double3 orientation;
  double k;
  double c;
  double c2;
  double c4;
  double c6;
  double c8;
  double c10;
} asphere_params;

typedef struct 
{
  double3 centre;
  double3 orientation;
  double curvatureRadius;
  double apertureRadius;
} sphere_params;

typedef struct 
{
  double3 root;
  double3 normal;
} plane_params;

#endif
