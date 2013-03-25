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

#ifndef MICROLENSARRAYITEM
#define MICROLENSARRAYITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "GeometryItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class MicroLensArrayItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class MicroLensArrayItem :
	public GeometryItem
{
	Q_OBJECT

	Q_ENUMS(MicroLensAptType);

	Q_PROPERTY(double microLensRadius READ getMicroLensRadius WRITE setMicroLensRadius DESIGNABLE true USER true);
	Q_PROPERTY(double microLensAptRad READ getMicroLensAptRad WRITE setMicroLensAptRad DESIGNABLE true USER true);
	Q_PROPERTY(double microLensPitch READ getMicroLensPitch WRITE setMicroLensPitch DESIGNABLE true USER true);
	Q_PROPERTY(MicroLensAptType microLensAptType READ getMicroLensAptType WRITE setMicroLensAptType DESIGNABLE true USER true);	
	Q_PROPERTY(double thickness READ getThickness WRITE setThickness DESIGNABLE true USER true);


public:
	enum MicroLensAptType {MICRORECTANGULAR, MICROELLIPTICAL, MICROUNKNOWN};

	MicroLensArrayItem(QString name="MicroLensArray", QObject *parent=0);
	~MicroLensArrayItem(void);

	// functions for property editor
	double getMicroLensRadius() const {return m_microLensRadius;};
	void setMicroLensRadius(const double in) {m_microLensRadius=in; emit itemChanged(m_index, m_index);};
	double getMicroLensAptRad() const {return m_microLensAptRad;};
	void setMicroLensAptRad(const double in) {m_microLensAptRad=in; emit itemChanged(m_index, m_index);};
	double getThickness() const {return m_thickness;};
	void setThickness(const double in) {m_thickness=in; emit itemChanged(m_index, m_index);};
	double getMicroLensPitch() const {return m_microLensPitch;};
	void setMicroLensPitch(const double in) {m_microLensPitch=in; emit itemChanged(m_index, m_index);};
	MicroLensAptType getMicroLensAptType() const {return m_microLensAptType;};
	void setMicroLensAptType(const MicroLensAptType in) {m_microLensAptType=in; emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void render(QMatrix4x4 &m, RenderOptions &options);
	Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);

	QString microLensAptTypeToString(MicroLensAptType in) const;
	MicroLensAptType stringToMicroLensAptType(QString in) const;

//	MaterialItem::MaterialType getMaterial() const {return m_materialType;};
//	void setMaterial(const MaterialItem::MaterialType type) {m_materialType=type;};

private:
	double m_microLensRadius;
	double m_microLensAptRad;
	double m_thickness;
	double m_microLensPitch;
	MicroLensAptType m_microLensAptType;

//	MaterialItem::MaterialType m_materialType;
};

}; //namespace macrosim

#endif