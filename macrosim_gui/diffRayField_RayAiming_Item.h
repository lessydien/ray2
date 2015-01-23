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

#ifndef DIFFRAYFIELDRAYAIMINGITEM
#define DIFFRAYFIELDRAYAIMINGITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "rayFieldItem.h"

using namespace macrosim;

namespace macrosim 
{

/** @class DiffRayField_RayAiming_Item
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class DiffRayField_RayAiming_Item :
	public RayFieldItem
{
	Q_OBJECT

        Q_PROPERTY(Vec3d initialTarget READ getInitialTarget WRITE setInitialTarget DESIGNABLE true USER true);	

public:

	DiffRayField_RayAiming_Item(QString name="DiffRayField_RayAiming", QObject *parent=0);
	~DiffRayField_RayAiming_Item(void);

	// functions for property editor
	const Vec3d getInitialTarget()  {return m_initialTarget;};
	void setInitialTarget(const Vec3d in) {m_initialTarget=in;};

	bool signalDataChanged();

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);
	void render(QMatrix4x4 &m, RenderOptions &options);
	void renderVtk(vtkSmartPointer<vtkRenderer> renderer);

private:
    Vec3d m_initialTarget;

}; // class DiffRayField_RayAiming_Item


}; //namespace macrosim

#endif