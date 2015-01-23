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


#include "CustomTypes.h"

#include "Vec3fProperty.h"
#include "Vec3dProperty.h"
#include "Vec2dProperty.h"
#include "Vec2iProperty.h"
#include "Vec9siProperty.h"
#include "Vec9dProperty.h"

#include <QPropertyEditor/Property.h>

Vec3d concatenateTilts(Vec3d tilt1, Vec3d tilt2)
{
	tilt1=tilt1/360*(2*PI);
	tilt2=tilt2/360*(2*PI);
	Mat3x3d Mx1(1,0,0, 0,cos(tilt1.X),-sin(tilt1.X), 0,sin(tilt1.X),cos(tilt1.X));
	Mat3x3d My1(cos(tilt1.Y),0,sin(tilt1.Y), 0,1,0, -sin(tilt1.Y),0,cos(tilt1.Y));
	Mat3x3d Mz1(cos(tilt1.Z),-sin(tilt1.Z),0, sin(tilt1.Z),cos(tilt1.Z),0, 0,0,1);
	Mat3x3d Mxy1=Mx1*My1;
	Mat3x3d M1=Mxy1*Mz1;

	Mat3x3d Mx2(1,0,0, 0,cos(tilt2.X),-sin(tilt2.X), 0,sin(tilt2.X),cos(tilt2.X));
	Mat3x3d My2(cos(tilt2.Y),0,sin(tilt2.Y), 0,1,0, -sin(tilt2.Y),0,cos(tilt2.Y));
	Mat3x3d Mz2(cos(tilt2.Z),-sin(tilt2.Z),0, sin(tilt2.Z),cos(tilt2.Z),0, 0,0,1);
	Mat3x3d Mxy2=Mx2*My2;
	Mat3x3d M2=Mxy2*Mz2;

	Mat3x3d Mges=M1*M2;

	// now calc angles from resulting matrix
	double theta1, theta2, phi, psi, theta;
	if ((Mges.m13 != 1) && (Mges.m13 != -1))
	{
		theta1=asin(Mges.m13);
		theta2=PI-theta1;
		if (abs(cos(theta1)) > abs(cos(theta2)))
			theta=theta1;
		else
			theta=theta2;

		phi=-atan2(Mges.m23/cos(theta), Mges.m33/cos(theta));
		psi=-atan2(Mges.m12/cos(theta), Mges.m11/cos(theta));
	}
	else
	{
		psi=0;
		if (Mges.m13==-1)
		{
			theta=-PI/2;
			phi=-psi+atan2(-Mges.m21,-Mges.m31);
		}
		else
		{
			theta=PI/2;
			phi=psi+atan2(Mges.m21,Mges.m31);
		}
	}

	Vec3d angles=Vec3d(phi, theta, psi);
	return angles/(2*PI)*360;
}

namespace CustomTypes
{
	void registerTypes()
	{
		static bool registered = false;
		if (!registered)
		{
			qRegisterMetaType<Vec3f>("Vec3f");
			qRegisterMetaType<Vec3d>("Vec3d");
			qRegisterMetaType<Vec2d>("Vec2d");
			qRegisterMetaType<Vec2i>("Vec2i");
			qRegisterMetaType<Vec9si>("Vec9si");
			qRegisterMetaType<Vec9d>("Vec9d");
			registered = true;
		}
	}

	Property* createCustomProperty(const QString& name, QObject* propertyObject, Property* parent)
	{
		int userType = propertyObject->property(qPrintable(name)).userType();

		if (userType == QMetaType::type("Vec3f"))
			return new Vec3fProperty(name, propertyObject, parent);
		if (userType == QMetaType::type("Vec3d"))
			return new Vec3dProperty(name, propertyObject, parent);
		if (userType == QMetaType::type("Vec2d"))
			return new Vec2dProperty(name, propertyObject, parent);
		if (userType == QMetaType::type("Vec2i"))
			return new Vec2iProperty(name, propertyObject, parent);
		if (userType == QMetaType::type("Vec9si"))
			return new Vec9siProperty(name, propertyObject, parent);
		if (userType == QMetaType::type("Vec9d"))
			return new Vec9dProperty(name, propertyObject, parent);
		return new Property(name, propertyObject, parent);
	}
}
