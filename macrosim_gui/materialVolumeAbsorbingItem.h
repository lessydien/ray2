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

#ifndef MATERIALVOLUMEABSORBINGINGITEM
#define MATERIALVOLUMEABSORBINGINGITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "MaterialItem.h"


namespace macrosim 
{

/** @class MaterialVolumeAbsorbingItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class MaterialVolumeAbsorbingItem :
	public MaterialItem
{
	Q_OBJECT
		Q_PROPERTY(double asorbCoeff READ getAbsorbCoeff WRITE setAbsorbCoeff DESIGNABLE true USER true);	

public:
	
	MaterialVolumeAbsorbingItem(QString name="MaterialVolumeAbsorbing", QObject *parent=0);
	~MaterialVolumeAbsorbingItem(void);

	// functions for property editor
	const double getAbsorbCoeff()  {return m_absorbCoeff;};
	void setAbsorbCoeff(const double in) {m_absorbCoeff=in;};


	bool writeToXML(QDomDocument &document, QDomElement &root) const ;
	bool readFromXML(const QDomElement &node);

	double getTransparency() {return 0.5;};

private:
	double m_absorbCoeff;
};

}; //namespace macrosim

#endif