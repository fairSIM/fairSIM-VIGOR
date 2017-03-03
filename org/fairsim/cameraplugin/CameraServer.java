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
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.DataFormatException;
import org.fairsim.cameraplugin.CameraPlugin.CameraException;
import org.fairsim.controller.AbstractServer;
import org.fairsim.controller.ServerGui;
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class CameraServer extends AbstractServer {

    CameraController cc;

    private CameraServer(ServerGui gui, CameraController cc) throws IOException {
        super(gui);
        this.cc = cc;
    }

    private String setRoi(int x, int y, int width, int height) {
        //cc.stopAcquisition();
        try {
            String output;
            try {
                cc.setRoi(x, y, width, height);
                output = "ROI successfully set";
            } catch (DataFormatException ex) {
                int[] roi = cc.getRoi();
                output = "ROI was set to: (" + roi[0] + ", " + roi[1] + ", " + roi[2] + ", " + roi[3] + ")";
            }
            gui.showText(output);
            return output;
        } catch (CameraException ex) {
            return ex.toString();
        }
        
    }

    private String getRoi() {
        try {
            int[] roi = cc.getRoi();
            gui.showText("Transfering roi");
            return Tool.encodeArray("Transfering roi", roi);
        } catch (CameraException ex) {
            return ex.toString();
        }
    }

    private String setExposureTime(double time) {
        try {
            cc.setExposure(time);
            String output = "Exposure time was set to: " + cc.getExposure() + "ms";
            gui.showText(output);
            return output;
        } catch (CameraException ex) {
            return ex.toString();
        }
    }

    private String getExposureTime() {
        try {
            String output = "Transfering exposure time;" + cc.getExposure();
            gui.showText(output);
            return output;
        } catch (CameraException ex) {
            return ex.toString();
        }
    }

    private String getGroups() {
        CameraGroup[] groups = cc.getGroups();
        int len = groups.length;
        String[] groupStrings = new String[len];
        for (int i = 0; i < len; i++) {
            groupStrings[i] = groups[i].encode();
        }
        gui.showText("Transfering groups");
        return Tool.encodeArray("Transfering groups", groupStrings);
    }

    private String setConfig(int groupId, int configId) {
        try {
            cc.setConfig(groupId, configId);
            String output = "New config has been set";
            gui.showText(output);
            return output;
        } catch (CameraException ex) {
            return ex.toString();
        }
    }
    
    private String startAcquisition() {
        cc.startNetworkAcquisition();
        String output = "Acquisition started";
        gui.showText(output);
        return output;
    }
    
    private String stopAcquisition() {
        cc.stopAcquisition();
        String output = "Acquisition stopped";
        gui.showText(output);
        return output;
    }
    
    private String getStatus() {
        String[] status = new String[3];
        status[0] = String.valueOf(cc.fps);
        status[1] = String.valueOf(cc.queued);
        status[2] = String.valueOf(cc.sended);
        return Tool.encodeArray("Transfering status", status);
    }

    @Override
    protected void buildUpConnection() {
        cc.stopAcquisition();
    }

    @Override
    protected void buildDownConnection() {
        cc.stopAcquisition();
    }

    @Override
    protected String handleCommand(String input) {
        if (input.equals("get roi")) {
            return getRoi();
        } else if (input.equals("get exposure")) {
            return getExposureTime();
        } else if (input.equals("get groups")) {
            return getGroups();
        } else if (input.equals("get status")) {
            return getStatus();
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

    static CameraServer startCameraServer(CameraServerGui gui, CameraController cc) {
        try {
            CameraServer serverObject = new CameraServer(gui, cc);
            serverObject.start();
            return serverObject;
        } catch (IOException ex) {
            return null;
        }
    }
}
