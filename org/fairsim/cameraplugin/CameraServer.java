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

package org.fairsim.cameraplugin;

import java.io.IOException;
import org.fairsim.controller.AbstractServer;
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class CameraServer extends AbstractServer {

    CameraPlugin cp;

    private CameraServer(CameraServerGui cameraGui, CameraPlugin cp) throws IOException {
        super(cameraGui);
        this.cp = cp;
    }

    private String setRoi(int x, int y, int width, int height) {
        try {
            cp.stopAcquisition();
            cp.setRoi(x, y, width, height);
            int[] roi = cp.getRoi();
            return "ROI was set to: (" + roi[0] + ", " + roi[1] + ", " + roi[2] + ", " + roi[3] + ")";
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    private String getRoi() {
        try {
            int[] roi = cp.getRoi();
            return Tool.encodeArray("Transfering roi", roi);
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    private String setExposureTime(double time) {
        try {
            cp.setExposureTime(time);
            return "Exposure time was set to: " + cp.getExposureTime() + "ms";
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    private String getExposureTime() {
        try {
            return "Exposure time: " + cp.getExposureTime();
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    private String getGroups() {
        try {
            CameraGroup[] groups = cp.getGroups();
            int len = groups.length;
            String[] groupStrings = new String[len];
            for (int i = 0; i < len; i++) {
                groupStrings[i] = groups[i].encode();
            }
            return Tool.encodeArray("Transfering groups", groupStrings);
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    private String setConfig(int groupId, int configId) {
        try {
            cp.setConfig(groupId, configId);
            return "Config has been set";
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    @Override
    protected void buildUpConnection() {
    }

    @Override
    protected void buildDownConnection() {
    }

    @Override
    protected String handleCommand(String input) {
        System.out.println("Recived command: " + input);
        String serverOut = "---";
        if (input.equals("get roi")) {
            return this.getRoi();
        } else if (input.equals("get exposure")) {
            return this.getExposureTime();
        } else if (input.equals("get groups")) {
            return this.getGroups();
        } else {
            return "Camera server do not know what to do with '" + input + "'";
        }
    }

    static CameraServer startCameraServer(CameraServerGui gui, CameraPlugin cp) {
        try {
            CameraServer serverObject = new CameraServer(gui, cp);
            serverObject.start();
            return serverObject;
        } catch (IOException ex) {
            return null;
        }
    }
}
