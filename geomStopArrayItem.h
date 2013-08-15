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

#ifndef STOPARRAYITEM
#define STOPARRAYITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "GeometryItem.h"

#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>

using namespace macrosim;

namespace macrosim 
{

/** @class StopArrayItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class StopArrayItem :
	public GeometryItem
{
	Q_OBJECT

	Q_ENUMS(MicroStopType);

	Q_PROPERTY(Vec2d microStopRad READ getMicroStopRad WRITE setMicroStopRad DESIGNABLE true USER true);
	Q_PROPERTY(Vec2d microStopPitch READ getMicroStopPitch WRITE setMicroStopPitch DESIGNABLE true USER true);
	Q_PROPERTY(MicroStopType microStopType READ getMicroStopType WRITE setMicroStopType DESIGNABLE true USER true);	

	Q_CLASSINFO("microStopRad", "decimals=10;");
	Q_CLASSINFO("microStopPitch", "decimals=10;");

public:
	enum MicroStopType {MICROSTOP_RECTANGULAR, MICROSTOP_ELLIPTICAL, MICROSTOP_UNKNOWN};

	StopArrayItem(QString name="StopArray", QObject *parent=0);
	~StopArrayItem(void);

	// functions for property editor
	Vec2d getMicroStopRad() const {return m_microStopRad;};
	void setMicroStopRad(const Vec2d in) {m_microStopRad=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	Vec2d getMicroStopPitch() const {return m_microStopPitch;};
	void setMicroStopPitch(const Vec2d in) {m_microStopPitch=in; this->updateVtk(); emit itemChanged(m_index, m_index);};
	MicroStopType getMicroStopType() const {return m_microStopType;};
	void setMicroStopType(const MicroStopType in) {m_microStopType=in; this->updateVtk(); emit itemChanged(m_index, m_index);};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);

	Vec3f calcNormal(Vec3f vertex, Vec3f* neighbours, int nr);
	Vec3f calcNormal(Vec3f vertex);

	QString microStopTypeToString(MicroStopType in) const;
	MicroStopType stringToMicroStopType(QString in) const;


private:
	void updateVtk();

	vtkSmartPointer<vtkPolyData> m_pPolydata;
	vtkSmartPointer<vtkPolyDataMapper> m_pMapper;

	Vec2d m_microStopRad;
	Vec2d m_microStopPitch;
	MicroStopType m_microStopType;

};

}; //namespace macrosim

#endif