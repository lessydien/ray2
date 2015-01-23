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


#include "Vec9siProperty.h"
#include "CustomTypes.h"

#include <qregexp.h>

Vec9siProperty::Vec9siProperty(const QString& name /*= QString()*/, QObject* propertyObject /*= 0*/, QObject* parent /*= 0*/) : Property(name, propertyObject, parent)
{
	m_x1 = new Property("x1", this, this);
	m_x2 = new Property("x2", this, this);
	m_x3 = new Property("x3", this, this);
	m_x4 = new Property("x4", this, this);
	m_x5 = new Property("x5", this, this);
	m_x6 = new Property("x6", this, this);
	m_x7 = new Property("x7", this, this);
	m_x8 = new Property("x8", this, this);
	m_x9 = new Property("x9", this, this);
	setEditorHints("minimumX=0;maximumX=999999999;minimumY=0;maximumY=999999999;");
}

QVariant Vec9siProperty::value(int role) const
{
	QVariant data = Property::value();
	if (data.isValid() && role != Qt::UserRole)
	{
		switch (role)
		{
		case Qt::DisplayRole:
			return tr("[ %1, %2, %3, %4, %5 ,%6, %7, %8, %9]").arg(data.value<Vec9si>().X1).arg(data.value<Vec9si>().X2).arg(data.value<Vec9si>().X3).arg(data.value<Vec9si>().X4).arg(data.value<Vec9si>().X5).arg(data.value<Vec9si>().X6).arg(data.value<Vec9si>().X7).arg(data.value<Vec9si>().X8).arg(data.value<Vec9si>().X9);
		case Qt::EditRole:
			return tr("%1, %2, %3, %4, %5, %6, %7, %8, %9").arg(data.value<Vec9si>().X1).arg(data.value<Vec9si>().X2).arg(data.value<Vec9si>().X3).arg(data.value<Vec9si>().X4).arg(data.value<Vec9si>().X5).arg(data.value<Vec9si>().X6).arg(data.value<Vec9si>().X7).arg(data.value<Vec9si>().X8).arg(data.value<Vec9si>().X9);
		};
	}
	return data;
}

void Vec9siProperty::setValue(const QVariant& value)
{
	if (value.type() == QVariant::String)
	{
		QString v = value.toString();				
		QRegExp rx("([+-]?([0-9]+))");
		rx.setCaseSensitivity(Qt::CaseInsensitive);
		int count = 0;
		int pos = 0;
		int x1 = 0, x2 = 0, x3 = 0, x4 = 0, x5 = 0, x6 = 0, x7 = 0, x8 = 0, x9 = 0;
		while ((pos = rx.indexIn(v, pos)) != -1) 
		{
			if (count == 0)
				x1 = rx.cap(1).toInt();
			else if (count == 1)
				x2 = rx.cap(1).toInt();
			else if (count == 2)
				x3 = rx.cap(1).toInt();
			else if (count == 3)
				x4 = rx.cap(1).toInt();
			else if (count == 4)
				x5 = rx.cap(1).toInt();
			else if (count == 5)
				x6 = rx.cap(1).toInt();
			else if (count == 6)
				x7 = rx.cap(1).toInt();
			else if (count == 7)
				x8 = rx.cap(1).toInt();
			else if (count == 8)
				x9 = rx.cap(1).toInt();
			else if (count>8)
				break;
			++count;
			pos += rx.matchedLength();
		}
		m_x1->setProperty("x1", x1);
		m_x2->setProperty("x2", x2);
		m_x3->setProperty("x3", x3);
		m_x4->setProperty("x4", x4);
		m_x5->setProperty("x5", x5);
		m_x6->setProperty("x6", x6);
		m_x7->setProperty("x7", x7);
		m_x8->setProperty("x8", x8);
		m_x9->setProperty("x9", x9);
		Property::setValue(QVariant::fromValue(Vec9si(x1, x2, x3, x4, x5, x6, x7, x8, x9)));
	}
	else
		Property::setValue(value);
}

void Vec9siProperty::setEditorHints(const QString& hints)
{
	m_x1->setEditorHints(parseHints(hints, 'X1'));
	m_x2->setEditorHints(parseHints(hints, 'X2'));
	m_x3->setEditorHints(parseHints(hints, 'X3'));
	m_x4->setEditorHints(parseHints(hints, 'X4'));
	m_x5->setEditorHints(parseHints(hints, 'X5'));
	m_x6->setEditorHints(parseHints(hints, 'X6'));
	m_x7->setEditorHints(parseHints(hints, 'X7'));
	m_x8->setEditorHints(parseHints(hints, 'X8'));
	m_x9->setEditorHints(parseHints(hints, 'X9'));
}

int Vec9siProperty::x1() const
{
	return value().value<Vec9si>().X1;
}

void Vec9siProperty::setX1(int in)
{
	Property::setValue(QVariant::fromValue(Vec9si(in, x2(), x3(), x4(), x5(), x6(), x7(), x8(), x9())));
}

int Vec9siProperty::x2() const
{
	return value().value<Vec9si>().X2;
}

void Vec9siProperty::setX2(int in)
{
	Property::setValue(QVariant::fromValue(Vec9si(x1(),in, x3(), x4(), x5(), x6(), x7(), x8(), x9())));
}

int Vec9siProperty::x3() const
{
	return value().value<Vec9si>().X3;
}

void Vec9siProperty::setX3(int in)
{
	Property::setValue(QVariant::fromValue(Vec9si(x1(), x2(), in, x4(), x5(), x6(), x7(), x8(), x9())));
}

int Vec9siProperty::x4() const
{
	return value().value<Vec9si>().X4;
}

void Vec9siProperty::setX4(int in)
{
	Property::setValue(QVariant::fromValue(Vec9si(x1(), x2(), x3(), in, x5(), x6(), x7(), x8(), x9())));
}

int Vec9siProperty::x5() const
{
	return value().value<Vec9si>().X5;
}

void Vec9siProperty::setX5(int in)
{
	Property::setValue(QVariant::fromValue(Vec9si(x1(), x2(), x3(), x4(), in, x5(), x7(), x8(), x9())));
}

int Vec9siProperty::x6() const
{
	return value().value<Vec9si>().X6;
}

void Vec9siProperty::setX6(int in)
{
	Property::setValue(QVariant::fromValue(Vec9si(x1(), x2(), x3(), x4(), x5(), in, x7(), x8(), x9())));
}

int Vec9siProperty::x7() const
{
	return value().value<Vec9si>().X7;
}

void Vec9siProperty::setX7(int in)
{
	Property::setValue(QVariant::fromValue(Vec9si(x1(), x2(), x3(), x4(), x5(), x6(), in, x8(), x9())));
}

int Vec9siProperty::x8() const
{
	return value().value<Vec9si>().X8;
}

void Vec9siProperty::setX8(int in)
{
	Property::setValue(QVariant::fromValue(Vec9si(x1(), x2(), x3(), x4(), x5(), x6(), x7(), in, x9())));
}

int Vec9siProperty::x9() const
{
	return value().value<Vec9si>().X9;
}

void Vec9siProperty::setX9(int in)
{
	Property::setValue(QVariant::fromValue(Vec9si(x1(), x2(), x3(), x4(), x5(), x6(), x7(), x8(), in)));
}

QString Vec9siProperty::parseHints(const QString& hints, const QChar component )
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