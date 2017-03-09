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

import mmcorej.CMMCore;
import org.micromanager.api.ScriptInterface;

/**
 *
 * @author m.lachetta
 */
public class CameraPlugin implements org.micromanager.api.MMPlugin {

    public static String menuName = "Fast SIM Camera Controller";
    public static String tooltipDescription = "Micro manager plugin to controll the cameras of the fast SIM setup";
    private static final int ROILENGTH = 4;
    CMMCore mmc;
    CameraController cc;
    
    public void startSequenceAcquisition() throws CameraException {
        try {
            mmc.startContinuousSequenceAcquisition(1);
        } catch (Exception ex) {
            throw new CameraException("Starting Acquisition went wrong");
        }
    }
    
    public void stopSequenceAcquisition() throws CameraException {
        try {
            mmc.stopSequenceAcquisition();
        } catch (Exception ex) {
            throw new CameraException("Stopping Acquisition went wrong");
        }
    }
    
    public void setBuffer(int buffer) throws CameraException {
        try {
            mmc.setCircularBufferMemoryFootprint(buffer);
            mmc.initializeCircularBuffer();
        } catch (Exception ex) {
            throw new CameraException("Set buffer went wrong");
        }
    }
    
    public boolean isSequenceRunning() throws CameraException {
        try {
            return mmc.isSequenceRunning();
        } catch (Exception ex) {
            throw new CameraException("Getting sequence running went wrong");
        }
    }
    
    public int getRemainingImageCount() throws CameraException {
        try {
            return mmc.getRemainingImageCount();
        } catch (Exception ex) {
            throw new CameraException("Getting image count went wrong");
        }
    }
    
    public short[] getNextImage() throws CameraException {
        try {
            return (short[]) mmc.popNextImage();
        } catch (Exception ex) {
            throw new CameraException("Getting next image went wrong");
        }
    }
    
    public void sleepCam(int time) throws CameraException {
        try {
            mmc.sleep(time);
        } catch (Exception ex) {
            throw new CameraException("Sleeping went wrong");
        }
    }
    
    public void setROI(int x, int y, int w, int h) throws CameraException {
        try {;
            mmc.clearROI();
            mmc.setROI(x, y, w, h);
        } catch (Exception ex) {
            throw new CameraException("Set ROI went wrong");
        }
    }
    
    public int getImageWidth() throws CameraException {
        try {
            return (int) mmc.getImageWidth();
        } catch (Exception ex) {
            throw new CameraException("Getting width went wrong");
        }
    }
    
    public int getImageHeight() throws CameraException {
        try {
            return (int) mmc.getImageHeight();
        } catch (Exception ex) {
            throw new CameraException("Getting height went wrong");
        }
    }
    
    public int[] getRoi() throws CameraException {
        int[] x = new int[1];
        int[] y = new int[1];
        int[] w = new int[1];
        int[] h = new int[1];
        try {
            mmc.getROI(x, y, w, h);
        } catch (Exception ex) {
            throw new CameraException("Getting ROI went wrong");
        }
        int[] roi = new int[ROILENGTH];
        roi[0] = x[0];
        roi[1] = y[0];
        roi[2] = w[0];
        roi[3] = h[0];
        return roi;
    }
    
    public void setExposure(double time) throws CameraException {
        try {
            mmc.setExposure(time);
        } catch (Exception ex) {
            throw new CameraException("Set exposure time went wrong");
        }
    }
    
    public double getExposure() throws CameraException {
        try {
            return mmc.getExposure();
        } catch (Exception ex) {
            throw new CameraException("Getting exposure time went wrong");
        }
    }
    
    public void setConfig(String group, String config) throws CameraException {
        try {
            mmc.setConfig(group , config);
        } catch (Exception ex) {
            throw new CameraException("Set config went wrong");
        }
    }
    
    public String[] getConfigGroups() throws CameraException {
        try {
            return mmc.getAvailableConfigGroups().toArray();
        } catch (Exception ex) {
            throw new CameraException("Getting groups went wrong");
        }
    }
    
    public String[] getConfigs(String group) throws CameraException {
        try {
            return mmc.getAvailableConfigs(group).toArray();
        } catch (Exception ex) {
            throw new CameraException("Getting configs went wrong");
        }
    }
    
    class CameraException extends Exception {

        CameraException(String message) {
            super("Error: " + message);
            System.out.println(message);
        }
    }

    @Override
    public void dispose() {
    }

    @Override
    public void setApp(ScriptInterface si) {
        /*
        try {
            initPlugin(si);
            prepareAcquisition(768, 768, 512, 512);
            guiFrame.startButton.setEnabled(true);
            guiFrame.stopButton.setEnabled(false);
        } catch (Exception ex) {
            System.out.println(ex);
        }
        */
        
        try {
            mmc = si.getMMCore();
            cc = new CameraController(this);
        } catch (Exception ex) {
            si.showError(ex);
        }
    }

    @Override
    public void show() {
        System.out.println(menuName + " got opend via mirco manager");
    }

    @Override
    public String getDescription() {
        return "Micro manager plugin that handels cameras of the fast sim"
                + "setup at the university of bielefeld.";
    }

    @Override
    public String getInfo() {
        return getDescription();
    }

    @Override
    public String getVersion() {
        return "alpha-version 1.1";
    }

    @Override
    public String getCopyright() {
        return "there is no right to copy";
    }

}
