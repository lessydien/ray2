#ifndef GEOMETRYPRIMITIVE_H
  #define GEOMETRYPRIMITIVE_H

#include <optix.h>

typedef enum 
{
  ASPHERICALSURFTEST
} geometryPrimitive_type;

typedef struct
{
  double x, y, z;
}  double3Struct;

class GeometryPrimitive
{
  protected:
    int primitiveID;
    float boundingBox_max[3];
    float boundingBox_min[3];
    geometryPrimitive_type type;
	char path_to_ptx_boundingBox[512];
	char path_to_ptx_intersect[512];
	RTmaterial    material;

  public:
    int  getPrimitiveID(void);
    void setPrimitiveID(int ID);
    void setBoundingBox_min(float *box_min);
    float* getBoundingBox_min(void);
    void setBoundingBox_max(float* box_max);
    float* getBoundingBox_max(void);
    geometryPrimitive_type getPrimitiveType(void);
    void createOptiXBoundingBox( RTcontext &context, RTgeometry &geometry );
	void setPathToPtxIntersect(char* path);
	const char* getPathToPtxIntersect(void);
	void setPathToPtxBoundingBox(char* path);
	const char* getPathToPtxBoundingBox(void);
	void setType(geometryPrimitive_type type);
	geometryPrimitive_type getType(void);
	void setMaterial(RTmaterial &materialIn);
	RTmaterial getMaterial(void);
};

#endif

