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

#ifndef APERTUREARRAYITEM
#define APERTUREARRAYITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "GeometryItem.h"

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>

using namespace macrosim;

namespace macrosim 
{

/** @class ApertureArrayItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class ApertureArrayItem :
	public GeometryItem
{
	Q_OBJECT

	Q_ENUMS(MicroAptType);

	Q_PROPERTY(Vec2d microApertureRad READ getMicroAptRad WRITE setMicroAptRad DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d microAperturePitch READ getMicroAperturePitch WRITE setMicroAperturePitch DESIGNABLE true USER true);
	Q_PROPERTY(MicroAptType microApertureType READ getMicroAptType WRITE setMicroAptType DESIGNABLE true USER true);
	Q_CLASSINFO("microApertureRad", "decimals=10;");
	Q_CLASSINFO("microAperturePitch", "decimals=10;");


public:
	enum MicroAptType {MICROAPT_RECTANGULAR, MICROAPT_ELLIPTICAL, MICROAPT_UNKNOWN};

	ApertureArrayItem(QString name="ApertureArray", QObject *parent=0);
	~ApertureArrayItem(void);

	// functions for property editor
	Vec2d getMicroAptRad() const {return m_microAptRad;};
	void setMicroAptRad(const Vec2d in) {m_microAptRad=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	Vec2d getMicroAperturePitch() const {return m_microAptPitch;};
	void setMicroAperturePitch(const Vec2d in) {m_microAptPitch=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	MicroAptType getMicroAptType() const {return m_microAptType;};
	void setMicroAptType(const MicroAptType in) {m_microAptType=in; this->updateVtk(); emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);

	Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);
	Vec3f calcNormal(Vec3f vertex);

	QString microAptTypeToString(MicroAptType in) const;
	MicroAptType stringToMicroAptType(QString in) const;

//	MaterialItem::MaterialType getMaterial() const {return m_materialType;};
//	void setMaterial(const MaterialItem::MaterialType type) {m_materialType=type;};

private:
	void updateVtk();

	vtkSmartPointer<vtkPolyData> m_pPolydata;
	vtkSmartPointer<vtkPolyDataMapper> m_pMapper;

	Vec2d m_microAptRad;
	Vec2d m_microAptPitch;
	MicroAptType m_microAptType;

	float calcZCoordinate(float x, float y, float lensHeightMax, Vec3f* z);

//	MaterialItem::MaterialType m_materialType;
};

}; //namespace macrosim

#endif