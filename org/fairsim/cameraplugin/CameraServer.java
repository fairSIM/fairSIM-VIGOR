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
import java.util.zip.DataFormatException;
import org.fairsim.cameraplugin.CameraPlugin.CameraException;
import org.fairsim.controller.AbstractServer;
import org.fairsim.utils.Tool;

/**
 * Class for the server on camera side to control cameras from the reconstruction
 * computer directly out of fairSIM
 * @author m.lachetta
 */
public class CameraServer extends AbstractServer {

    CameraController cc;

    /**
     * Constructor
     * @param gui GUI of this server
     * @param cc controller of this camera
     * @throws IOException of connecting went wrong
     */
    private CameraServer(ServerGui gui, CameraController cc) throws IOException {
        super(gui, 32323);
        this.cc = cc;
    }

    /**
     * sets the region of interest of the camera
     * @param x upper left corner x value
     * @param y upper left corner y value
     * @param width acquire image width
     * @param height acquire image height
     * @param imageSize squared sending image size
     * @return answer for the client
     */
    private String setRoi(int x, int y, int width, int height, int imageSize) {
        //cc.stopAcquisition();
        try {
            String output;
            try {
                cc.setRoi(x, y, width, height, imageSize);
                output = "ROI successfully set";
            } catch (DataFormatException ex) {
                int[] roi = cc.getRoi();
                output = "ROI was set to: (" + roi[0] + ", " + roi[1] + ", " + roi[2] + ", " + roi[3] + ", " + roi[4] + ")";
            }
            gui.showText(output);
            return output;
        } catch (CameraException ex) {
            return ex.toString();
        }
    }
    
    /**
     * sets the 512 region of interest of the camera
     * @return answer for the client
     */
    private String setBigRoi() {
        int[] roi;
        try {
            cc.setBigRoi();
            roi = cc.getRoi();
            String output = "ROI was set to: (" + roi[0] + ", " + roi[1] + ", " + roi[2] + ", " + roi[3] + ", " + roi[4] + ")";
            gui.showText(output);
            return output;
        } catch (CameraException | DataFormatException ex) {
            return ex.toString();
        } 
    }
    
    /**
     * sets the 256 region of interest of the camera
     * @return answer for the client
     */
    private String setSmallRoi() {
        int[] roi;
        try {
            cc.setSmallRoi();
            roi = cc.getRoi();
            String output = "ROI was set to: (" + roi[0] + ", " + roi[1] + ", " + roi[2] + ", " + roi[3] + ", " + roi[4] + ")";
            gui.showText(output);
            return output;
        } catch (CameraException | DataFormatException ex) {
            return ex.toString();
        } 
    }

    /**
     * 
     * @return answer for the client, the encoded region of interest
     */
    private String getRoi() {
        try {
            int[] roi = cc.getRoi();
            gui.showText("Transfering roi");
            return Tool.encodeArray("Transfering roi", roi);
        } catch (CameraException ex) {
            return ex.toString();
        }
    }

    /**
     * sets the exposure time of the camera
     * @param time exposure time in milliseconds
     * @return answer for the client
     */
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

    /**
     * 
     * @return answer for the client, exposure time in milliseconds
     */
    private String getExposureTime() {
        try {
            String output = "Transfering exposure time;" + cc.getExposure();
            gui.showText(output);
            return output;
        } catch (CameraException ex) {
            return ex.toString();
        }
    }

    /**
     * 
     * @return answer for the client, encoded array containing groups with configs
     */
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

    /**
     * sets a specified config of this camera
     * @param groupId id of the group of configs
     * @param configId id of the config
     * @return answer for the client
     */
    private String setConfig(int groupId, int configId) {
        try {
            cc.setConfig(groupId, configId);
            String output = "New config has been set";
            gui.showText(output);
            return output;
        } catch (CameraException | IllegalArgumentException ex) {
            return ex.toString();
        }
    }

    /**
     * starts acquiring images
     * @return answer for the client
     */
    private String startAcquisition() {
        cc.startNetworkAcquisition();
        String output = "Acquisition started";
        gui.showText(output);
        return output;
    }

    /**
     * stops acquiring images
     * @return answer for the client
     */
    private String stopAcquisition() {
        cc.stopAcquisition();
        String output = "Acquisition stopped";
        gui.showText(output);
        return output;
    }

    /**
     * 
     * @return answer for the client, encoded array containing fps and sending status
     */
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
            int s = Integer.parseInt(sRoi[4]);
            return setRoi(x, y, w, h, s);
        } else if (input.equals("set big roi")) {
            return setBigRoi();
        } else if (input.equals("set small roi")) {
            return setSmallRoi();
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

    /**
     * creates and starts a CameraServer
     * @param gui GUI of the CameraServer
     * @param cc CameraController of the CameraServer
     * @return the CameraServer
     */
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
