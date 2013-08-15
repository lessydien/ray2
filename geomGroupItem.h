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

#ifndef GEOMGROUPITEM
#define GEOMGROUPITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "geometryItem.h"
#include "MiscItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class GeomGroupItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class GeomGroupItem :
	public MiscItem
{
	Q_OBJECT

	Q_PROPERTY(AccelerationType accelerationType READ getAccType WRITE setAccType DESIGNABLE true USER true);

	Q_ENUMS(AccelerationType);

public:

	enum AccelerationType {NOACCEL, SBVH, BVH, MEDIANBVH, LBVH, TRIANGLEKDTREE};

	GeomGroupItem(QString name="GeometryGroup", QObject *parent=0);
	~GeomGroupItem(void);

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void render(QMatrix4x4 &m, RenderOptions &options);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);

	AccelerationType getAccType() const {return m_accelType;};
	void setAccType(const AccelerationType in) {m_accelType=in;};

	QModelIndex hasActor(void *actor) const;

	QString accelerationTypeToString(const AccelerationType in) const;
	AccelerationType stringToAccelerationType(const QString in) const;
	void setRenderOptions(RenderOptions options);

	void removeFromView(vtkSmartPointer<vtkRenderer> renderer);

private:
	AccelerationType m_accelType;

};

}; //namespace macrosim

#endif