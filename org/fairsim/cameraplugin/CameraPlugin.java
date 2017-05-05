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
 * Class that implements the micro manager plugin
 * @author m.lachetta
 */
public class CameraPlugin implements org.micromanager.api.MMPlugin {

    public static String menuName = "Fast-SIM Camera Controller";
    public static String tooltipDescription = "Micro manager plugin to controll the cameras of the fast SIM setup";
    private static final int ROILENGTH = 4;
    CMMCore mmc;
    CameraController cc;
    
    /**
     * starts camera acquisition
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public void startSequenceAcquisition() throws CameraException {
        try {
            mmc.startContinuousSequenceAcquisition(1);
        } catch (Exception ex) {
            throw new CameraException("Starting Acquisition went wrong");
        }
    }
    
    /**
     * stops camera acquisition
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public void stopSequenceAcquisition() throws CameraException {
        try {
            mmc.stopSequenceAcquisition();
        } catch (Exception ex) {
            throw new CameraException("Stopping Acquisition went wrong");
        }
    }
    
    /**
     * set the camera/micro manager buffer to a specific size
     * @param buffer size of the buffer
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public void setBuffer(int buffer) throws CameraException {
        try {
            mmc.setCircularBufferMemoryFootprint(buffer);
            mmc.initializeCircularBuffer();
        } catch (Exception ex) {
            throw new CameraException("Set buffer went wrong");
        }
    }
    
    /**
     * 
     * @return if the camera is acquiring
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public boolean isSequenceRunning() throws CameraException {
        try {
            return mmc.isSequenceRunning();
        } catch (Exception ex) {
            throw new CameraException("Getting sequence running went wrong");
        }
    }
    
    /**
     * 
     * @return number of images in the buffer
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public int getRemainingImageCount() throws CameraException {
        try {
            return mmc.getRemainingImageCount();
        } catch (Exception ex) {
            throw new CameraException("Getting image count went wrong");
        }
    }
    
    /**
     * 
     * @return the next image from the buffer
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public short[] getNextImage() throws CameraException {
        try {
            return (short[]) mmc.popNextImage();
        } catch (Exception ex) {
            throw new CameraException("Getting next image went wrong");
        }
    }
    
    /**
     * Waits (blocks the calling thread) for specified time in milliseconds.
     * @param time waiting time
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public void sleepCam(int time) throws CameraException {
        try {
            mmc.sleep(time);
        } catch (Exception ex) {
            throw new CameraException("Sleeping went wrong");
        }
    }
    
    /**
     * sets the region of interest of the camera
     * @param x upper left corner x value
     * @param y upper left corner y value
     * @param w image width
     * @param h image height
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public void setROI(int x, int y, int w, int h) throws CameraException {
        try {;
            mmc.clearROI();
            mmc.setROI(x, y, w, h);
        } catch (Exception ex) {
            throw new CameraException("Set ROI went wrong");
        }
    }
    
    /**
     * 
     * @return width of images which will be acquire
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public int getImageWidth() throws CameraException {
        try {
            return (int) mmc.getImageWidth();
        } catch (Exception ex) {
            throw new CameraException("Getting width went wrong");
        }
    }
    
    /**
     * 
     * @return height of images which will be acquire
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public int getImageHeight() throws CameraException {
        try {
            return (int) mmc.getImageHeight();
        } catch (Exception ex) {
            throw new CameraException("Getting height went wrong");
        }
    }
    
    /**
     * 
     * @return an array with the region of interest of the camera (x,y,width,height)
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
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
    
    /**
     * sets the exposure time (not directly global exposure) of the camera
     * @param time exposure time in milliseconds
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public void setExposure(double time) throws CameraException {
        try {
            mmc.setExposure(time);
        } catch (Exception ex) {
            throw new CameraException("Set exposure time went wrong");
        }
    }
    
    /**
     * 
     * @return the exposure time of the camera
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public double getExposure() throws CameraException {
        try {
            return mmc.getExposure();
        } catch (Exception ex) {
            throw new CameraException("Getting exposure time went wrong");
        }
    }
    
    /**
     * sets a specified config of the camera
     * @param group group of config
     * @param config config to be set
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public void setConfig(String group, String config) throws CameraException {
        try {
            mmc.setConfig(group , config);
        } catch (Exception ex) {
            throw new CameraException("Set config went wrong");
        }
    }
    
    /**
     * 
     * @return array of groups of config
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public String[] getConfigGroups() throws CameraException {
        try {
            return mmc.getAvailableConfigGroups().toArray();
        } catch (Exception ex) {
            throw new CameraException("Getting groups went wrong");
        }
    }
    
    /**
     * 
     * @param group the group of configs
     * @return the config array of a specified group
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public String[] getConfigs(String group) throws CameraException {
        try {
            return mmc.getAvailableConfigs(group).toArray();
        } catch (Exception ex) {
            throw new CameraException("Getting configs went wrong");
        }
    }
    
    /**
     * Exception class to handle errors in communication with micro manager and 
     * cameras
     */
    class CameraException extends Exception {

        CameraException(String message) {
            super("Error: " + message);
            System.out.println(message);
        }
    }

    /**
     * called when micro manager is closing
     */
    @Override
    public void dispose() {
    }

    /**
     * called by micro while starting this plugin
     * @param si ScriptInterface of micro manager
     */
    @Override
    public void setApp(ScriptInterface si) {
        try {
            mmc = si.getMMCore();
            cc = new CameraController(this);
        } catch (Exception ex) {
            si.showError(ex);
        }
    }

    /**
     * called by micro manager after setApp
     * @see setApp(ScriptInterface si)
     */
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
        return "version 1";
    }

    @Override
    public String getCopyright() {
        return "GPL licence";
    }

}
