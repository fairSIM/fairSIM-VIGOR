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

import org.fairsim.cameraplugin.CameraGroup;
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class CameraClient extends AbstractClient {
    private String channelName;
    private CameraGroup[] groups;

    public CameraClient(String serverAdress, int serverPort, ClientGui clientGui, String channelName) {
        super(serverAdress, serverPort, clientGui);
        this.channelName = channelName;
    }

    @Override
    protected void handleServerAnswer(String answer) {
        if (answer.startsWith("Transfering groups")) {
            String[] groupStrings = Tool.decodeArray(answer);
            int len = groupStrings.length;
            groups = new CameraGroup[len];
            for (int i = 0; i < len; i++) {
                groups[i] = new CameraGroup(groupStrings[i]);
            }
        } else if (answer.startsWith("Error: ")) {
            //TODO
            clientGui.showText(answer);
        } else {
            clientGui.showText(answer);
        }
    }
    
}
