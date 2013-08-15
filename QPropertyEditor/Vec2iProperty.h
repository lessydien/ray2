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


#ifndef VEC2IPROPERTY_H_
#define VEC2IPROPERTY_H_

#include <QPropertyEditor/Property.h>

/**
 * A custom property for Vec2i 
 * 
 * This class provides a QLineEdit editor for all two values of a Vec2i struct (x, y)
 * and also contains two QSpinBox instances to manipulate each value separately
 */
class Vec2iProperty : public Property
{
	Q_OBJECT
	Q_PROPERTY(unsigned int x READ x WRITE setX DESIGNABLE true USER true)
	Q_PROPERTY(unsigned int y READ y WRITE setY DESIGNABLE true USER true)

public:
	Vec2iProperty(const QString& name = QString(), QObject* propertyObject = 0, QObject* parent = 0);

	QVariant value(int role = Qt::UserRole) const;
	virtual void setValue(const QVariant& value);

	void setEditorHints(const QString& hints);

	unsigned int x() const;
	void setX(unsigned int x);

	unsigned int y() const;
	void setY(unsigned int y);

private:
	QString parseHints(const QString& hints, const QChar component);

	Property*	m_x;
	Property*	m_y;
};
#endif
