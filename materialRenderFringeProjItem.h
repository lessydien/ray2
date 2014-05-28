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

#ifndef MATERIALRENDERFRINGEPROJTITEM
#define MATERIALRENDERFRINGEPROJTITEM

#include <qicon.h>

#include "QPropertyEditor/CustomTypes.h"
#include "AbstractItem.h"
#include "GeometryItem.h"
#include "MaterialItem.h"


namespace macrosim 
{

/** @class MaterialRenderFringeProjItem
*   @brief class for visualizing objects from MacroSim
*   
*   The AddInModel supplies a widget showing the available objects from MacroSim with their name, filename, version and so on.
*/
class MaterialRenderFringeProjItem :
	public MaterialItem
{
	Q_OBJECT

    Q_ENUMS(FringeType);
    Q_ENUMS(FringeOrientation);

    Q_PROPERTY(double power READ getPower WRITE setPower DESIGNABLE true USER true);
    Q_PROPERTY(Vec3d pupilRoot READ getPupilRoot WRITE setPupilRoot DESIGNABLE true USER true);
    Q_PROPERTY(Vec3d pupilTilt READ getPupilTilt WRITE setPupilTilt DESIGNABLE true USER true);
    Q_PROPERTY(Vec2d pupilAptRad READ getPupilAptRad WRITE setPupilAptRad DESIGNABLE true USER true);
    Q_PROPERTY(FringeType fringeType READ getFringeType WRITE setFringeType DESIGNABLE true USER true);
    Q_PROPERTY(FringeOrientation fringeOrientation READ getFringeOrientation WRITE setFringeOrientation DESIGNABLE true USER true);
    Q_PROPERTY(double fringePeriod READ getFringePeriod WRITE setFringePeriod DESIGNABLE true USER true);
    Q_PROPERTY(double fringePhase READ getFringePhase WRITE setFringePhase DESIGNABLE true USER true);
    Q_PROPERTY(int nrBits READ getNrBits WRITE setNrBits DESIGNABLE true USER true);
    Q_PROPERTY(int codeNr READ getCodeNr WRITE setCodeNr DESIGNABLE true USER true);
    Q_PROPERTY(double power READ getPower WRITE setPower DESIGNABLE true USER true);
    

public:

    enum FringeType {GRAYCODE, SINUS, UNKNOWN};
    enum FringeOrientation {FO_X, FO_Y};
	
	MaterialRenderFringeProjItem(double power=1, FringeType fringeType=SINUS, FringeOrientation orientation=FO_X, double fringePeriod=10, int nrBits=4, int codeNr=1, double fringePhase=0, QString name="MaterialRenderFringeProj", QObject *parent=0);
	~MaterialRenderFringeProjItem(void);

	// functions for property editor
    void setPower(const double in) {m_power=in; emit itemChanged(m_index, m_index);};
    double getPower() const {return m_power;}
    void setPupilRoot(const Vec3d in) {m_pupilRoot=in; emit itemChanged(m_index, m_index);};
    Vec3d getPupilRoot() const {return m_pupilRoot;}
    void setPupilTilt(const Vec3d in) {m_pupilTilt=in; emit itemChanged(m_index, m_index);};
    Vec3d getPupilTilt() const {return m_pupilTilt;}
    void setPupilAptRad(const Vec2d in) {m_pupilAptRad=in; emit itemChanged(m_index, m_index);};
    Vec2d getPupilAptRad() const {return m_pupilAptRad;}
    void setFringeType(const FringeType in) {m_fringeType=in; emit itemChanged(m_index, m_index);};
    FringeType getFringeType() const {return m_fringeType;};
    void setFringePeriod(const double in) {m_fringePeriod=in; emit itemChanged(m_index, m_index);};
    double getFringePeriod() const {return m_fringePeriod;};
    FringeOrientation getFringeOrientation() const {return m_fringeOrientation;};
    void setFringeOrientation(const FringeOrientation in) {m_fringeOrientation=in; emit itemChanged(m_index, m_index);};
    void setNrBits(const int in) {m_nrBits=in; emit itemChanged(m_index, m_index);};
    int getNrBits() const {return m_nrBits;};
    void setCodeNr(const int in) {m_codeNr=in; emit itemChanged(m_index, m_index);};
    int getCodeNr() const {return m_codeNr;};
    void setFringePhase(const double in) {m_fringePhase=in; emit itemChanged(m_index, m_index);};
    double getFringePhase() const {return m_fringePhase;};

	bool writeToXML(QDomDocument &document, QDomElement &root) const;
	bool readFromXML(const QDomElement &node);

	double getTransparency() {return 0.5;};


private:
    QString fringeTypeToString(FringeType fringeType) const;
    FringeType strToFringeType(QString str) const;
    QString fringeOrientationToString(FringeOrientation fringeOrientation) const;
    FringeOrientation strToFringeOrientation(QString str) const;

	Vec3d m_pupilRoot;
	Vec3d m_pupilTilt;
    Vec2d m_pupilAptRad;
    FringeType m_fringeType;
    FringeOrientation m_fringeOrientation;
    double m_fringePeriod;
    int m_nrBits;
    int m_codeNr;
    double m_power;
    double m_fringePhase;
};

}; //namespace macrosim

#endif