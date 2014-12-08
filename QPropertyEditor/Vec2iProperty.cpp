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


#include "Vec2iProperty.h"
#include "CustomTypes.h"

#include <qregexp.h>

Vec2iProperty::Vec2iProperty(const QString& name /*= QString()*/, QObject* propertyObject /*= 0*/, QObject* parent /*= 0*/) : Property(name, propertyObject, parent)
{
	m_x = new Property("x", this, this);
	m_y = new Property("y", this, this);
	setEditorHints("minimumX=0;maximumX=999999999;minimumY=0;maximumY=999999999;");
}

QVariant Vec2iProperty::value(int role) const
{
	QVariant data = Property::value();
	if (data.isValid() && role != Qt::UserRole)
	{
		switch (role)
		{
		case Qt::DisplayRole:
			return tr("[ %1, %2]").arg(data.value<Vec2i>().X).arg(data.value<Vec2i>().Y);
		case Qt::EditRole:
			return tr("%1, %2").arg(data.value<Vec2i>().X).arg(data.value<Vec2i>().Y);
		};
	}
	return data;
}

void Vec2iProperty::setValue(const QVariant& value)
{
	if (value.type() == QVariant::String)
	{
		QString v = value.toString();				
		QRegExp rx("([0-9]+)");
		rx.setCaseSensitivity(Qt::CaseInsensitive);
		int count = 0;
		int pos = 0;
		unsigned int x = 0, y = 0;
		while ((pos = rx.indexIn(v, pos)) != -1) 
		{
			if (count == 0)
				x = rx.cap(1).toUInt();
			else if (count == 1)
				y = rx.cap(1).toUInt();
			else if (count > 1)
				break;
			++count;
			pos += rx.matchedLength();
		}
		m_x->setProperty("x", x);
		m_y->setProperty("y", y);
		Property::setValue(QVariant::fromValue(Vec2i(x, y)));
	}
	else
		Property::setValue(value);
}

void Vec2iProperty::setEditorHints(const QString& hints)
{
	m_x->setEditorHints(parseHints(hints, 'X'));
	m_y->setEditorHints(parseHints(hints, 'Y'));
}

unsigned int Vec2iProperty::x() const
{
	return value().value<Vec2i>().X;
}

void Vec2iProperty::setX(unsigned int x)
{
	Property::setValue(QVariant::fromValue(Vec2i(x, y())));
}

unsigned int Vec2iProperty::y() const
{
	return value().value<Vec2i>().Y;
}

void Vec2iProperty::setY(unsigned int y)
{
	Property::setValue(QVariant::fromValue(Vec2i(x(), y)));
}

QString Vec2iProperty::parseHints(const QString& hints, const QChar component )
{
	QRegExp rx(QString("(.*)(")+component+QString("{1})(=\\s*)(.*)(;{1})"));
	rx.setMinimal(true);
	int pos = 0;
	QString componentHints;
	while ((pos = rx.indexIn(hints, pos)) != -1) 
	{
		// cut off additional front settings (TODO create correct RegExp for that)
		if (rx.cap(1).lastIndexOf(';') != -1)			
			componentHints += QString("%1=%2;").arg(rx.cap(1).remove(0, rx.cap(1).lastIndexOf(';')+1)).arg(rx.cap(4));
		else
			componentHints += QString("%1=%2;").arg(rx.cap(1)).arg(rx.cap(4));
		pos += rx.matchedLength();
	}
	return componentHints;
}