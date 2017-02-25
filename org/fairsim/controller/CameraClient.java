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
    String channelName;
    ControllerClientGui clientGui;
    int[] rois;
    double exposure;
    private CameraGroup[] groups;

    public CameraClient(String serverAdress, int serverPort, ControllerClientGui clientGui, String channelName) {
        super(serverAdress, serverPort, clientGui);
        this.channelName = channelName;
        this.clientGui = clientGui;
    }
    
    String[] getGroupArray() {
        int len = groups.length;
        String[] s = new String[len];
        for (int i = 0; i < len; i++) {
            s[i] = groups[i].getNmae();
        }
        return s;
    }
    
    String[] getConfigArray(int groupId) {
        return groups[groupId].getConfigArray();
    }

    @Override
    protected void handleServerAnswer(String answer) {
        if (answer.startsWith("Transfering roi")) {
            String[] sRois = Tool.decodeArray(answer);
            int len = sRois.length;
            rois = new int[len];
            for (int i = 0; i < len; i++) {
                rois[i] = Integer.parseInt(sRois[i]);
            }
        }
        else if (answer.startsWith("Transfering groups")) {
            String[] groupStrings = Tool.decodeArray(answer);
            int len = groupStrings.length;
            groups = new CameraGroup[len];
            for (int i = 0; i < len; i++) {
                groups[i] = new CameraGroup(groupStrings[i]);
            }
        } else if (answer.startsWith("Transfering exposure time")) {
            exposure = Double.parseDouble(answer.split(";")[1]);
        } else if (answer.startsWith("Acquisition")) {
            //do nothing
        } else if (answer.startsWith("Error: ")) {
            clientGui.handleCamError(answer, this);
        } else {
            clientGui.showText(answer);
        }
    }
    
}
