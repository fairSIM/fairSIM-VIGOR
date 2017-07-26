/*
This file is part of Free Analysis and Interactive Reconstruction
for Structured Illumination Microscopy (fairSIM).

fairSIM is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

fairSIM is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with fairSIM.  If not, see <http://www.gnu.org/licenses/>
*/

package org.fairsim.controller;

/**
 * interface to control a slm
 * @author m.lachetta
 */
public abstract class SlmController {

    /**
     * sets a new selected running order
     *
     * @param ro new running order
     * @return String-output-Stream
     */
    abstract String setRo(int ro);

    /**
     * activates the selected running order
     *
     * @return String-output-Stream
     */
    abstract String activateRo();

    /**
     * deactivates the current running order
     *
     * @return out String-output-Stream
     */
    abstract String deactivateRo();

    /**
     * Sends an array of running orders of the SLM to the client
     *
     * @return out String-output-Stream
     */
    abstract String getRoList();

    /**
     * Reboots the SLM <br>
     * not really recommended to use this
     *
     * @return out String-output-Stream
     */
    abstract int rebootSlm();

    /**
     * Opens the connection between Server and SLM
     *
     * @return out String-output-Stream
     */
    abstract String connectSlm();

    /**
     * Closes the connection between Server and SLM
     *
     * @return out String-output-Stream
     */
    abstract String disconnectSlm();
}
