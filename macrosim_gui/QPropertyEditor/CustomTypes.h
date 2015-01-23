// *************************************************************************************************
//
// This code is part of the Sample Application to demonstrate the use of the QPropertyEditor library.
// It is distributed as public domain and can be modified and used by you without any limitations.
// 
// Your feedback is welcome!
//
// Author: Volker Wiendl
// Enum enhancement by Roman alias banal from qt-apps.org
// *************************************************************************************************

#ifndef CUSTOMTYPES_H_
#define CUSTOMTYPES_H_

#include <qvariant.h>
#include <math.h>

#ifndef PI
#define PI ((double)3.141592653589793238462643383279502884197169399375105820)
#endif

class Property;
class QObject;

struct Vec2i
{
	Vec2i() : X(0), Y(0) {} 
	Vec2i(unsigned int x, unsigned int y) : X(x), Y(y) {}
	unsigned int X, Y;

	bool operator == (const Vec2i& other) const {return X == other.X && Y == other.Y ;} 
	bool operator != (const Vec2i& other) const {return X != other.X || Y != other.Y ;} 
	Vec2i operator + (const Vec2i& other) const {return Vec2i(X+other.X,Y+other.Y);}
	Vec2i operator - (const Vec2i& other) const {return Vec2i(X-other.X,Y-other.Y);}
	double operator * (const Vec2i& other) const {return X*other.X+Y*other.Y;}
	Vec2i operator * (const unsigned int s) const {return Vec2i(X*s,Y*s);}
};
Q_DECLARE_METATYPE(Vec2i)

struct Vec9si
{
	Vec9si() : X1(0), X2(0), X3(0), X4(0), X5(0), X6(0), X7(0), X8(0), X9(0) {} 
	Vec9si(int x1, int x2, int x3, int x4, int x5, int x6, int x7, int x8, int x9) : X1(x1), X2(x2), X3(x3), X4(x4), X5(x5), X6(x6), X7(x7), X8(x8), X9(x9) {}
	int X1, X2, X3, X4, X5, X6, X7, X8, X9;

	bool operator == (const Vec9si& other) const {return X1 == other.X1 && X2 == other.X2 && X3 == other.X3 && X4 == other.X4 && X5 == other.X5 && X6 == other.X6 && X7 == other.X7 && X8 == other.X8 && X9 == other.X9 ;} 
	bool operator != (const Vec9si& other) const {return X1 != other.X1 || X2 != other.X2 || X3 != other.X3 || X4 != other.X4 || X5 != other.X5 || X6 != other.X6 || X7 != other.X7 || X8 != other.X8 || X9 != other.X9 ;} 
	Vec9si operator + (const Vec9si& other) const {return Vec9si(X1+other.X1,X2+other.X2,X3+other.X3,X4+other.X4,X5+other.X5,X6+other.X6,X7+other.X7,X8+other.X8,X9+other.X9);}
	Vec9si operator - (const Vec9si& other) const {return Vec9si(X1-other.X1,X2-other.X2,X3-other.X3,X4-other.X4,X5-other.X5,X6-other.X6,X7-other.X7,X8-other.X8,X9-other.X9);}
};
Q_DECLARE_METATYPE(Vec9si)

struct Vec9d
{
	Vec9d() : X1(0.0), X2(0.0), X3(0.0), X4(0.0), X5(0.0), X6(0.0), X7(0.0), X8(0.0), X9(0.0) {} 
	Vec9d(double x1, double x2, double x3, double x4, double x5, double x6, double x7, double x8, double x9) : X1(x1), X2(x2), X3(x3), X4(x4), X5(x5), X6(x6), X7(x7), X8(x8), X9(x9) {}
	double X1, X2, X3, X4, X5, X6, X7, X8, X9;

	bool operator == (const Vec9d& other) const {return X1 == other.X1 && X2 == other.X2 && X3 == other.X3 && X4 == other.X4 && X5 == other.X5 && X6 == other.X6 && X7 == other.X7 && X8 == other.X8 && X9 == other.X9 ;} 
	bool operator != (const Vec9d& other) const {return X1 != other.X1 || X2 != other.X2 || X3 != other.X3 || X4 != other.X4 || X5 != other.X5 || X6 != other.X6 || X7 != other.X7 || X8 != other.X8 || X9 != other.X9 ;} 
	Vec9d operator + (const Vec9d& other) const {return Vec9d(X1+other.X1,X2+other.X2,X3+other.X3,X4+other.X4,X5+other.X5,X6+other.X6,X7+other.X7,X8+other.X8,X9+other.X9);}
	Vec9d operator - (const Vec9d& other) const {return Vec9d(X1-other.X1,X2-other.X2,X3-other.X3,X4-other.X4,X5-other.X5,X6-other.X6,X7-other.X7,X8-other.X8,X9-other.X9);}
};
Q_DECLARE_METATYPE(Vec9d)

struct Vec3f
{
	Vec3f() : X(0.0f), Y(0.0f), Z(0.0f) {} 
	Vec3f(float x, float y, float z) : X(x), Y(y), Z(z) {}
	float X, Y, Z;

	bool operator == (const Vec3f& other) const {return X == other.X && Y == other.Y && Z == other.Z;} 
	bool operator != (const Vec3f& other) const {return X != other.X || Y != other.Y || Z != other.Z;} 
	Vec3f operator - (const Vec3f& other) const {return Vec3f(X-other.X,Y-other.Y,Z-other.Z);}
	Vec3f operator + (const Vec3f& other) const {return Vec3f(X+other.X,Y+other.Y,Z+other.Z);}
	float operator * (const Vec3f& other) const {return X*other.X+Y*other.Y+Z*other.Z;}
	Vec3f operator * (const float s) const {return Vec3f(X*s,Y*s,Z*s);}
	Vec3f operator / (const float s) const {return Vec3f(X/s,Y/s,Z/s);}


};
Q_DECLARE_METATYPE(Vec3f)

struct Mat3x3f
{
	Mat3x3f() : m11(0.0), m12(0.0), m13(0.0),  m21(0.0), m22(0.0), m23(0.0),  m31(0.0), m32(0.0), m33(0.0) {}
	Mat3x3f(float i11, float i12, float i13,  float i21, float i22, float i23,  float i31, float i32, float i33) : m11(i11), m12(i12), m13(i13),  m21(i21), m22(i22), m23(i23),  m31(i31), m32(i32), m33(i33) {}

	float m11, m12, m13, m21, m22, m23, m31, m32, m33;

	Mat3x3f operator * (const Mat3x3f other) const 
	{
		Mat3x3f mOut;
		mOut.m11 = m11*other.m11+m12*other.m21+m13*other.m31;
		mOut.m12 = m11*other.m12+m12*other.m22+m13*other.m32;
		mOut.m13 = m11*other.m13+m12*other.m23+m13*other.m33;
		mOut.m21 = m21*other.m11+m22*other.m21+m23*other.m31;
		mOut.m22 = m21*other.m12+m22*other.m22+m23*other.m32;
		mOut.m23 = m21*other.m13+m22*other.m23+m23*other.m33;
		mOut.m31 = m31*other.m11+m32*other.m21+m33*other.m31;
		mOut.m32 = m31*other.m12+m32*other.m22+m33*other.m32;
		mOut.m33 = m31*other.m13+m32*other.m23+m33*other.m33;
		return mOut;	
	};

	Vec3f operator * (const Vec3f other) const 
	{
		Vec3f b;
		b.X = m11*other.X+m12*other.Y+m13*other.Z;
		b.Y = m21*other.X+m22*other.Y+m23*other.Z;
		b.Z = m31*other.X+m32*other.Y+m33*other.Z;
		return b;	
	};
};

inline Vec3f cross(Vec3f& a, Vec3f b)
{
	return Vec3f(a.Y*b.Z-a.Z*b.Y, a.Z*b.X-a.X*b.Z, a.X*b.Y-a.Y*b.X);
}

struct Vec3d
{
	Vec3d() : X(0.0), Y(0.0), Z(0.0) {} 
	Vec3d(double x, double y, double z) : X(x), Y(y), Z(z) {}
	double X, Y, Z;

	bool operator == (const Vec3d& other) const {return X == other.X && Y == other.Y && Z == other.Z;} 
	bool operator != (const Vec3d& other) const {return X != other.X || Y != other.Y || Z != other.Z;} 
	Vec3d operator + (const Vec3d& other) const {return Vec3d(X+other.X,Y+other.Y,Z+other.Z);}
	Vec3d operator - (const Vec3d& other) const {return Vec3d(X-other.X,Y-other.Y,Z-other.Z);}
	double operator * (const Vec3d& other) const {return X*other.X+Y*other.Y+Z*other.Z;}
	Vec3d operator * (const double s) const {return Vec3d(X*s,Y*s,Z*s);}
	Vec3d operator / (const double s) const {return Vec3d(X/s,Y/s,Z/s);}
};
Q_DECLARE_METATYPE(Vec3d)

struct Vec2d
{
	Vec2d() : X(0.0), Y(0.0) {} 
	Vec2d(double x, double y) : X(x), Y(y) {}
	double X, Y;

	bool operator == (const Vec2d& other) const {return X == other.X && Y == other.Y ;} 
	bool operator != (const Vec2d& other) const {return X != other.X || Y != other.Y ;} 
	Vec2d operator + (const Vec2d& other) const {return Vec2d(X+other.X,Y+other.Y);}
	Vec2d operator - (const Vec2d& other) const {return Vec2d(X-other.X,Y-other.Y);}
	double operator * (const Vec2d& other) const {return X*other.X+Y*other.Y;}
	Vec2d operator * (const double s) const {return Vec2d(X*s,Y*s);}
	Vec2d operator / (const double s) const {return Vec2d(X/s,Y/s);}
};
Q_DECLARE_METATYPE(Vec2d)

struct Vec3l
{
	Vec3l() : X(0), Y(0), Z(0) {} 
	Vec3l(long x, long y, long z) : X(x), Y(y), Z(z) {}
	long X, Y, Z;

	bool operator == (const Vec3l& other) const {return X == other.X && Y == other.Y && Z == other.Z;} 
	bool operator != (const Vec3l& other) const {return X != other.X || Y != other.Y || Z != other.Z;} 
	Vec3l operator + (const Vec3l& other) const {return Vec3l(X+other.X,Y+other.Y,Z+other.Z);}
	Vec3l operator - (const Vec3l& other) const {return Vec3l(X-other.X,Y-other.Y,Z-other.Z);}
	long operator * (const Vec3l& other) const {return X*other.X+Y*other.Y+Z*other.Z;}
};
Q_DECLARE_METATYPE(Vec3l)


struct Mat3x3d
{
	Mat3x3d() : m11(0.0), m12(0.0), m13(0.0),  m21(0.0), m22(0.0), m23(0.0),  m31(0.0), m32(0.0), m33(0.0) {}
	Mat3x3d(double i11, double i12, double i13,  double i21, double i22, double i23,  double i31, double i32, double i33) : m11(i11), m12(i12), m13(i13),  m21(i21), m22(i22), m23(i23),  m31(i31), m32(i32), m33(i33) {}

	double m11, m12, m13, m21, m22, m23, m31, m32, m33;

	Mat3x3d operator * (const Mat3x3d other) const 
	{
		Mat3x3d mOut;
		mOut.m11 = m11*other.m11+m12*other.m21+m13*other.m31;// mOut.m11 = m1.m11*m2.m11+m1.m12*m2.m21+m1.m13*m2.m31;
		mOut.m12 = m11*other.m12+m12*other.m22+m13*other.m32;// mOut.m12 = m1.m11*m2.m12+m1.m12*m2.m22+m1.m13*m2.m32;
		mOut.m13 = m11*other.m13+m12*other.m23+m13*other.m33;// mOut.m13 = m1.m11*m2.m13+m1.m12*m2.m23+m1.m13*m2.m33;
		mOut.m21 = m21*other.m11+m22*other.m21+m23*other.m31;// mOut.m21 = m1.m21*m2.m11+m1.m22*m2.m21+m1.m23*m2.m31;
		mOut.m22 = m21*other.m12+m22*other.m22+m23*other.m32;// mOut.m22 = m1.m21*m2.m12+m1.m22*m2.m22+m1.m23*m2.m32;
		mOut.m23 = m21*other.m13+m22*other.m23+m23*other.m33;// mOut.m23 = m1.m21*m2.m13+m1.m22*m2.m23+m1.m23*m2.m33;
		mOut.m31 = m31*other.m11+m32*other.m21+m33*other.m31;// mOut.m31 = m1.m31*m2.m11+m1.m32*m2.m21+m1.m33*m2.m31;
		mOut.m32 = m31*other.m12+m32*other.m22+m33*other.m32;// mOut.m32 = m1.m31*m2.m12+m1.m32*m2.m22+m1.m33*m2.m32;
		mOut.m33 = m31*other.m13+m32*other.m23+m33*other.m33;// mOut.m33 = m1.m31*m2.m13+m1.m32*m2.m23+m1.m33*m2.m33;
		return mOut;	
	};

	Vec3d operator * (const Vec3d other) const 
	{
		Vec3d b;
		b.X = m11*other.X+m12*other.Y+m13*other.Z;
		b.Y = m21*other.X+m22*other.Y+m23*other.Z;
		b.Z = m31*other.X+m32*other.Y+m33*other.Z;
		return b;	
	};
};

inline void rotateVec3d(Vec3d *in, Vec3d tilt) 
{
	Vec3d tmp;
	tilt=tilt/360*(2*PI);
	Mat3x3d Mx(1,0,0, 0,cos(tilt.X),-sin(tilt.X), 0,sin(tilt.X),cos(tilt.X));
	Mat3x3d My(cos(tilt.Y),0,sin(tilt.Y), 0,1,0, -sin(tilt.Y),0,cos(tilt.Y));
	Mat3x3d Mz(cos(tilt.Z),-sin(tilt.Z),0, sin(tilt.Z),cos(tilt.Z),0, 0,0,1);
	Mat3x3d Mxy=Mx*My;
	Mat3x3d M=Mxy*Mz;
	tmp=M*(*in);
	*in=tmp; 
};

inline Vec3f rotateVec(Vec3f& vec, Vec3d tilt)
{
	Mat3x3f Mx=Mat3x3f(1,0,0, 0,cos(tilt.X),-sin(tilt.X), 0,sin(tilt.X),cos(tilt.X));
	Mat3x3f My=Mat3x3f(cos(tilt.X),0,sin(tilt.Y), 0,1,0, -sin(tilt.Y),0,cos(tilt.Y));
	Mat3x3f Mz=Mat3x3f(cos(tilt.Z),-sin(tilt.Z),0, sin(tilt.Z),cos(tilt.Z),0, 0,0,1);
	Mat3x3f Mxy=Mx*My;
	Mat3x3f M=Mxy*Mz;
	return M*vec;
}

Vec3d concatenateTilts(Vec3d tilt1, Vec3d tilt2);

namespace CustomTypes
{
	void registerTypes();
	Property* createCustomProperty(const QString& name, QObject* propertyObject, Property* parent);
}
#endif
