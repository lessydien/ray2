/***********************************************************************
 This file is part of ITO-MacroSim.

    ITO-MacroSim is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ITO-MacroSim is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
************************************************************************/

#ifndef DETECTORITEM
#define DETECTORITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "AbstractItem.h"
//#include <QtOpenGL\qglfunctions.h>
//////#include "glut.h"
#include "DataObject/dataobj.h"

using namespace macrosim;

typedef struct test
{
	int x;
	test() : x(1)
	{
	}
	test& test::operator=(const test& in)
	{
		if (this!=&in)
		{
			this->x=in.x;
		}
		return *this;
	}
};


namespace macrosim 
{

/** @class FieldItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class DetectorItem :
	public AbstractItem
{
	Q_OBJECT

	Q_PROPERTY(DetType detType READ getDetType WRITE setDetType DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d apertureHalfWidth READ getApertureHalfWidth WRITE setApertureHalfWidth DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d tilt READ getTilt WRITE setTilt DESIGNABLE true USER true);
	Q_PROPERTY(Vec3d root READ getRoot WRITE setRoot DESIGNABLE true USER true);
	Q_PROPERTY(DetOutFormat outFormat READ getDetOutFormat WRITE setDetOutFormat DESIGNABLE true USER true);
	Q_PROPERTY(QString FileName READ getFileName WRITE setFileName DESIGNABLE true USER true);

	Q_ENUMS(DetOutFormat);
	Q_ENUMS(DetType);

public:

	enum DetOutFormat {MAT, TEXT};
	enum DetType{INTENSITY, FIELD, RAYDATA, UNDEFINED};

	DetectorItem(QString name="name", DetType type=UNDEFINED, QObject *parent=0);
	~DetectorItem(void);

	// functions for property editor
	const DetType getDetType()  {return m_detType;};
	void setDetType(const DetType in) {m_detType=in;};
	const Vec2d getApertureHalfWidth()  {return m_apertureHalfWidth;};
	void setApertureHalfWidth(const Vec2d in) {m_apertureHalfWidth=in;};
	const Vec3d getTilt()  {return m_tilt;};
	void setTilt(const Vec3d in) {m_tilt=in;};
	const DetOutFormat getDetOutFormat()  {return m_detOutFormat;};
	void setDetOutFormat(const DetOutFormat in) {m_detOutFormat=in;};
	void setResultField(ito::DataObject in) {m_resultField=in;};
	ito::DataObject getResultField() const {return m_resultField;};
	void setFileName(const QString in) {m_fileName=in;};
	const QString getFileName() const {return m_fileName;};
	Vec3d getRoot() const {return m_root;};
	void setRoot(const Vec3d root) {m_root=root;};

	virtual bool writeToXML(QDomDocument &document, QDomElement &root) const;
	virtual bool readFromXML(const QDomElement &node);

private:
	DetOutFormat m_detOutFormat;
	DetType m_detType;
	Vec3d m_tilt;
	Vec3d m_root;
	Vec2d m_apertureHalfWidth;
	ito::DataObject m_resultField;
	QString m_fileName;

public slots:
	void simulationFinished(ito::DataObject field);

};

}; //namespace macrosim

#endif