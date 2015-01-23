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


#ifndef VEC2DPROPERTY_H_
#define VEC2DPROPERTY_H_

#include <QPropertyEditor/Property.h>

/**
 * A custom property for Vec3d 
 * 
 * This class provides a QLineEdit editor for all three values of a Vec3d struct (x, y, z)
 * and also contains three QDoubleSpinBox instances to manipulate each value separately
 */
class Vec2dProperty : public Property
{
	Q_OBJECT
	Q_PROPERTY(double x READ x WRITE setX DESIGNABLE true USER true)
	Q_PROPERTY(double y READ y WRITE setY DESIGNABLE true USER true)

public:
	Vec2dProperty(const QString& name = QString(), QObject* propertyObject = 0, QObject* parent = 0);

	QVariant value(int role = Qt::UserRole) const;
	virtual void setValue(const QVariant& value);

	void setEditorHints(const QString& hints);

	double x() const;
	void setX(double x);

	double y() const;
	void setY(double y);

private:
	QString parseHints(const QString& hints, const QChar component);

	Property*	m_x;
	Property*	m_y;
};
#endif
