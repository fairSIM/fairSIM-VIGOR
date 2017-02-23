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
import org.fairsim.controller.Utilities;

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
    
    private String setRIO(int x, int y, int width, int height) {
        try {
            cp.stopAcquisition();
            cp.setROI(x, y, width, height);
            int[] rio = cp.getRIO();
            return "RIO was set to: (" + rio[0] + ", " + rio[1] + ", " + rio[2] + ", " + rio[3] + ")";
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }
    
    private String getRIO() {
        try {
            int[] rio = cp.getRIO();
            String serverOut = "Transfering RIO";
            for (int output : rio) {
                serverOut += ";" + output;
            }
            return serverOut;
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
            return Double.toString(cp.getExposureTime());
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
            return Utilities.encodeArray("Transfering groups", groupStrings);
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }
    
    private String setConfig(int groupId, int configId){
        try {
            cp.setConfig(groupId, configId);
            return "Config has been set";
        } catch (CameraPlugin.CameraException ex) {
            return ex.toString();
        }
    }

    @Override
    protected void buildUpConnection() {}

    @Override
    protected void buildDownConnection() {}

    @Override
    protected String handleCommand(String string) throws IOException {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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
