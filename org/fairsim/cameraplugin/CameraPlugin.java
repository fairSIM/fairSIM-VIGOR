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

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import mmcorej.CMMCore;
import mmcorej.TaggedImage;
import org.fairsim.utils.Tool;
import org.json.JSONObject;
import org.micromanager.api.ScriptInterface;

/**
 * Class that implements the micro manager plugin
 * @author m.lachetta
 */
public class CameraPlugin implements org.micromanager.api.MMPlugin {

    public static String menuName = "fairSIM Camera Controller";
    public static String tooltipDescription = "Micro manager plugin to controll the cameras with fairSIM";
    private static final int ROILENGTH = 4;
    CMMCore mmc;
    CameraController cc;
    String[] cams;
    
    public int getChannelCount() {
        return cams.length;
    }
    
    /**
     * starts camera acquisition
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public void startSequenceAcquisition() throws CameraException {
        try {
            for (String cam : cams) {
                mmc.setCameraDevice(cam);
                mmc.startContinuousSequenceAcquisition(1);
            }
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
            for (String cam : cams) {
                mmc.stopSequenceAcquisition(cam);
            }
        } catch (Exception ex) {
            ex.printStackTrace();
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
            boolean output = false;
            for (String cam : cams) {
                output = output || mmc.isSequenceRunning(cam);
            }
            return output;
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
    public ChanneldImage getNextImage() throws CameraException {
        try {
            TaggedImage img = mmc.popNextTaggedImage();
            
            // get channel idx
            JSONObject md = img.tags;
            
            /*
            Iterator<String> keys = md.keys();
            while(keys.hasNext()) {
                System.out.println(keys.next());
            }
            */
            ////System.out.println("IDS uEye-ReadoutTime " + md.get("IDS uEye-ReadoutTime"));
            //System.out.println("uEye-Timestamp " + md.get("uEye-Timestamp"));
            //System.out.println("uEye-rawStamp " + md.get("uEye-rawStamp"));
            ////System.out.println("ElapsedTime-ms " + md.get("ElapsedTime-ms"));
            
            //get channel index
            int chIdx = -1;
            String cName = md.getString("Camera");
            for (int i = 0; i < cams.length; i++) {
                if (cName.equals(cams[i])) {
                    chIdx = i;
                    break;
                }
            }
            if (chIdx == -1) throw new CameraException("No channel found");

            // get pixels
            short[] pix;
            try {
                pix = (short[]) img.pix;
            } catch (ClassCastException ex) {
                byte[] bytes = (byte[]) img.pix;
                pix = new short[bytes.length];
                for (int i = 0; i < bytes.length; i++) {
                    pix[i] = bytes[i];
                }
            }
            
            //get timestamp
            long timestamp = -1;
            try {
                timestamp = (long) (Long.valueOf(md.getString("uEye-rawStamp").split("\\[", 2)[1].split("\\]")[0]) * 0.1);
            } catch (Exception ex) {
                timestamp = Tool.decodeBcdTimestamp(pix);
            }
            if (chIdx == -1) throw new CameraException("No timestamp found");
            
            return new ChanneldImage(chIdx, timestamp, pix);
        } catch (Exception ex) {
            ex.printStackTrace();
            throw new CameraException("Getting next image went wrong: " + ex);
        }
    }
    
    static class ChanneldImage {
        int chIdx;
        long timestamp;
        short[] img;
        
        ChanneldImage(int chIdx, long timestamp, short[] img) {
            this.chIdx = chIdx;
            this.timestamp = timestamp;
            this.img = img;
        }
    }
    
    /**
     * Waits (blocks the calling thread) for specified time in milliseconds.
     * @param time waiting time
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public void sleep(int time) throws CameraException {
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
        try {
            for (String cam : cams) {
                mmc.setCameraDevice(cam);
                mmc.clearROI();
                mmc.setROI(x, y, w, h);
            }
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
        int[][][] roi = new int [ROILENGTH][cams.length][1];

        for (int i = 0; i < cams.length; i++) {
            try {
                mmc.setCameraDevice(cams[i]);
                mmc.getROI(roi[0][i], roi[1][i], roi[2][i], roi[3][i]);
                for (int r = 0; r < ROILENGTH; r++) {
                    if(roi[r][0][0] != roi[r][i][0]) throw new CameraException("Rois of cams unequal");
                }
            } catch (Exception ex) {
                throw new CameraException("Getting ROI went wrong");
            }
        }
        
        int[] roiOut = new int[ROILENGTH];
        for (int i = 0; i < ROILENGTH; i++) {
            roiOut[i] = roi[i][0][0];
        }
        
        return roiOut;
    }
    
    /**
     * sets the exposure time (not directly global exposure) of the camera
     * @param time exposure time in milliseconds
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if anything
     * went wrong
     */
    public void setExposure(double time) throws CameraException {
        try {
            for (String cam : cams) {
                mmc.setExposure(cam, time);
            }
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
            double[] exposure = new double[cams.length];
            for (int i = 0; i < cams.length; i++) {
                mmc.setCameraDevice(cams[i]);
                exposure[i] = mmc.getExposure();
                if (exposure[0] != exposure[i]) throw new CameraException("Exposure time of cams unequal");
            }
            return exposure[0];
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
            
            // init multi camera devices
            String[] devices = mmc.getLoadedDevicesOfType(mmcorej.DeviceType.CameraDevice).toArray();
            
            boolean multiFound = false;
            List<String> camList = new LinkedList<String>();
            for (String device : devices) {
                mmc.setCameraDevice(device);
                String name = mmc.getDeviceName(device);
                String library = mmc.getDeviceLibrary(device);
                if (name.equals("Multi Camera") && library.equals("Utilities")) {
                    if (multiFound) throw new CameraException("Found more than 1 multi cam adapter");
                    multiFound = true;
                } else {
                    camList.add(device);
                }
            }
            cams = new String[camList.size()];
            cams = camList.toArray(cams);
            if (cams.length <= 1 && multiFound) throw new CameraException("Found multi cam only");
            
            cc = new CameraController(this);
        } catch (Exception ex) {
            si.showError(ex);
            ex.printStackTrace();
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
