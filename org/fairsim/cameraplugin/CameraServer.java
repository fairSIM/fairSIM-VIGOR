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
            String output = "ROI was set to: (" + roi[0] + ", " + roi[1] + ", " + roi[2] + ", " + roi[3] + ")";
            gui.showText(output);
            return output;
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    private String getRoi() {
        try {
            int[] roi = cp.getRoi();
            gui.showText("Transfering roi");
            return Tool.encodeArray("Transfering roi", roi);
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    private String setExposureTime(double time) {
        try {
            cp.setExposureTime(time);
            String output = "Exposure time was set to: " + cp.getExposureTime() + "ms";
            gui.showText(output);
            return output;
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    private String getExposureTime() {
        try {
            String output = "Transfering exposure time;" + cp.getExposureTime();
            gui.showText(output);
            return output;
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
            gui.showText("Transfering groups");
            return Tool.encodeArray("Transfering groups", groupStrings);
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    private String setConfig(int groupId, int configId) {
        try {
            cp.setConfig(groupId, configId);
            String output = "New config has been set";
            gui.showText(output);
            return output;
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }
    
    private String startAcquisition() {
        cp.startAcquisition();
        String output = "Acquisition started";
        gui.showText(output);
        return output;
    }
    
    private String stopAcquisition() {
        cp.stopAcquisition();
        String output = "Acquisition stopped";
        gui.showText(output);
        return output;
    }

    @Override
    protected void buildUpConnection() {
    }

    @Override
    protected void buildDownConnection() {
    }

    @Override
    protected String handleCommand(String input) {
        if (input.equals("get roi")) {
            return getRoi();
        } else if (input.equals("get exposure")) {
            return getExposureTime();
        } else if (input.equals("get groups")) {
            return getGroups();
        } else if (input.startsWith("set roi")) {
            String[] sRoi = Tool.decodeArray(input);
            int x = Integer.parseInt(sRoi[0]);
            int y = Integer.parseInt(sRoi[1]);
            int w = Integer.parseInt(sRoi[2]);
            int h = Integer.parseInt(sRoi[3]);
            return setRoi(x, y, w, h);
        } else if (input.startsWith("set exposure")) {
            String exposureString = input.split(";")[1];
            double exposureDouble = Double.parseDouble(exposureString);
            return setExposureTime(exposureDouble);
        } else if (input.startsWith("set config")) {
            String[] stringIds = Tool.decodeArray(input);
            int groupId = Integer.parseInt(stringIds[0]);
            int configId = Integer.parseInt(stringIds[1]);
            return setConfig(groupId, configId);
        } else if (input.equals("start")) {
            return startAcquisition();
        } else if (input.equals("stop")) {
            return stopAcquisition();
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
