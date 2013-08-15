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


#ifndef VEC9DPROPERTY_H_
#define VEC9DPROPERTY_H_

#include <QPropertyEditor/Property.h>

/**
 * A custom property for Vec2i 
 * 
 * This class provides a QLineEdit editor for all 9 values of a Vec9si struct (x1, x2, x3, x4, x5, x6, x7, x8, x9)
 * and also contains 9 QSpinBox instances to manipulate each value separately
 */
class Vec9dProperty : public Property
{
	Q_OBJECT
	Q_PROPERTY(double x1 READ x1 WRITE setX1 DESIGNABLE true USER true)
	Q_PROPERTY(double x2 READ x2 WRITE setX2 DESIGNABLE true USER true)
	Q_PROPERTY(double x3 READ x3 WRITE setX3 DESIGNABLE true USER true)
	Q_PROPERTY(double x4 READ x4 WRITE setX4 DESIGNABLE true USER true)
	Q_PROPERTY(double x5 READ x5 WRITE setX5 DESIGNABLE true USER true)
	Q_PROPERTY(double x6 READ x6 WRITE setX6 DESIGNABLE true USER true)
	Q_PROPERTY(double x7 READ x7 WRITE setX7 DESIGNABLE true USER true)
	Q_PROPERTY(double x8 READ x8 WRITE setX8 DESIGNABLE true USER true)
	Q_PROPERTY(double x9 READ x9 WRITE setX9 DESIGNABLE true USER true)

public:
	Vec9dProperty(const QString& name = QString(), QObject* propertyObject = 0, QObject* parent = 0);

	QVariant value(int role = Qt::UserRole) const;
	virtual void setValue(const QVariant& value);

	void setEditorHints(const QString& hints);

	double x1() const;
	void setX1(double x1);
	double x2() const;
	void setX2(double x2);
	double x3() const;
	void setX3(double x3);
	double x4() const;
	void setX4(double x4);
	double x5() const;
	void setX5(double x5);
	double x6() const;
	void setX6(double x6);
	double x7() const;
	void setX7(double x7);
	double x8() const;
	void setX8(double x8);
	double x9() const;
	void setX9(double x9);

private:
	QString parseHints(const QString& hints, const QChar component);

	Property*	m_x1;
	Property*	m_x2;
	Property*	m_x3;
	Property*	m_x4;
	Property*	m_x5;
	Property*	m_x6;
	Property*	m_x7;
	Property*	m_x8;
	Property*	m_x9;
};
#endif
