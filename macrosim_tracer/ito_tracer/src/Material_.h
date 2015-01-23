#ifndef MATERIAL_H
  #define MATERIAL_H

#include <optix.h>
#include "optix_math_new.h"
#include "rayData.h"


typedef enum 
{
  MAT_ERROR,
  MAT_NO_ERROR
} MaterialError;

class Material
{
  protected:
    char path_to_ptx[512];

  public:
    virtual MaterialError createOptiXInstance(RTcontext context, RTgeometryinstance &instance, int index);
	void setPathToPtx(char* path);
	char* getPathToPtx(void);
	virtual void hit(rayStruct &ray, double3 normal);

};

#endif

