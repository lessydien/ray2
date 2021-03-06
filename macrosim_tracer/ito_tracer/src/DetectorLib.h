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

/**\file DetectorLib.h
* \brief header file that includes all the detectors defined in the application. If you define a new detector that's where you need to include the header to make it visible to the complete application
* 
*           
* \author Mauch
*/

#ifndef DETECTORLIB_H
  #define DETECTORLIB_H

// include the individual materials
#include "Detector_Raydata.h"
#include "Detector_Intensity.h"
#include "Detector_Field.h"
#include "Detector_PhaseSpace.h"
#include "Detector_VolumeIntensity.h"
#include "Parser_XML.h"

class DetectorFab
{
protected:

public:

	DetectorFab()
	{

	}
	~DetectorFab()
	{

	}

	bool createDetInstFromXML(xml_node &node, vector<Detector*> &detVec) const;
};

#endif
