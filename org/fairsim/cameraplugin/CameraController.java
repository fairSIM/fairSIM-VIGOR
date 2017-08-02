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

import java.awt.Color;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.DataFormatException;
import org.fairsim.cameraplugin.CameraPlugin.CameraException;
import org.fairsim.transport.ImageSender;
import org.fairsim.transport.ImageWrapper;
import org.fairsim.utils.Conf;
import org.fairsim.utils.Tool;

/**
 * Class to control a camera via micro manager
 * @author Mario
 */
public class CameraController {

    private final CameraPlugin cp;
    private final CameraServerGui gui;
    private AcquisitionThread acquisitionThread;
    private final int channel;
    private final List<String> sendIps;
    private final CameraGroup[] groups;
    private static final int CAMBUFFER = 1000;
    private static final int FPSCOUNTS = 59;
    private int[] roi;
    private final int[] bigRoi, smallRoi;
    private int imageWidth;
    private int imageHeight;
    private int sendImageSize;
    private boolean mirrored;
    private final ImageSender isend;
    double  fps;
    boolean queued;
    boolean sended;
    
    /**
     * Constructor for this instance
     * @param cp CameraPlugin for communication with micro manager
     * @throws IOException if reading from .xlm went wrong
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if communication
     * with micro manager went wrong
     */
    CameraController(CameraPlugin cp) throws IOException, CameraException {
        this.cp = cp;
        isend = new ImageSender();
        sendIps = new ArrayList<>();
        
        // readin config from .xlm
        String filename = Tool.getFile(System.getProperty("user.home") + "/documents/fastsim-camera.xml").getAbsolutePath();
        try {
            Conf.Folder cfg = Conf.loadFile(filename).r().cd("camera-settings");
            channel = cfg.getInt("Channel").val();
            String[] flags = cfg.getStr("Flags").val().split(" ");
            bigRoi = cfg.getInt("BigRoi").vals();
            smallRoi = cfg.getInt("SmallRoi").vals();
            String[] ips = cfg.getStr("SendIps").val().split(" ");
            mirrored = false;
            for (String flag : flags) {
                if (flag.equals("mirrored")) {
                    mirrored = true;
                }
            }
            try {
                setRoi(bigRoi[0], bigRoi[1], bigRoi[2], bigRoi[3], bigRoi[4], true);
            } catch (DataFormatException ex) {
            }
            for (String ip : ips) {
                sendIps.add(ip);
            }
        } catch (Conf.SomeIOException ex) {
            throw new FileNotFoundException(filename);
        } catch (Conf.EntryNotFoundException ex) {
            throw new IOException("Entry not found: " + filename);
        }

        cp.setBuffer(CAMBUFFER);

        // initilizes groups & configs
        String[] grps = cp.getConfigGroups();
        groups = new CameraGroup[grps.length];
        for (int i = 0; i < grps.length; i++) {
            groups[i] = new CameraGroup(grps[i], cp.getConfigs(grps[i]));
        }

        // starts the GUI
        this.gui = new CameraServerGui(imageWidth, imageHeight, this);
    }
    
    int getChannel() {
        return channel;
    }

    /**
     * sets the region of interest of the camera
     * @param x upper left x value
     * @param y upper left y value
     * @param width image width
     * @param height image height
     * @param sendImageSize squared sending size
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if communication
     * with micro manager went wrong
     * @throws DataFormatException if ROI could not be set as preferred
     */
    void setRoi(int x, int y, int width, int height, int sendImageSize) throws CameraException, DataFormatException {
        setRoi(x, y, width, height, sendImageSize, false);
    }
    
    /**
     * 
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if communication
     * with micro manager went wrong
     * @throws DataFormatException if ROI could not be set as preferred
     */
    void setBigRoi() throws CameraException, DataFormatException {
        setRoi(bigRoi[0], bigRoi[1], bigRoi[2], bigRoi[3], bigRoi[4]);
    }
    
    /**
     * 
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if communication
     * with micro manager went wrong
     * @throws DataFormatException if ROI could not be set as preferred
     */
    void setSmallRoi() throws CameraException, DataFormatException {
        setRoi(smallRoi[0], smallRoi[1], smallRoi[2], smallRoi[3], smallRoi[4]);
    }

    /**
     * sets the region of interest of the camera
     * @param x upper left x value
     * @param y upper left y value
     * @param width image width
     * @param height image height
     * @param sendImageSize squared sending size
     * @param firstTime set this true if it is the first time as the ROI will be set
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if communication
     * with micro manager went wrong
     * @throws DataFormatException if ROI could not be set as preferred
     */
    private void setRoi(int x, int y, int width, int height, int sendSize, boolean firstTime) throws CameraException, DataFormatException {
        cp.stopSequenceAcquisition();
        cp.setROI(x, y, width, height);
        roi = cp.getRoi();
        imageWidth = cp.getImageWidth();
        imageHeight = cp.getImageHeight();
        if (imageWidth != roi[2] || imageHeight != roi[3]) {
            throw new RuntimeException("This should never happen!");
        }
        sendImageSize = sendSize;
        if (imageWidth < sendImageSize) sendImageSize = imageWidth;
        if (imageHeight < sendImageSize) sendImageSize = imageHeight;
        if (!firstTime) {
            gui.refreshView(imageWidth, imageHeight);
        }
        if (roi[0] != x || roi[1] != y || roi[2] != width || roi[3] != height) {
            throw new DataFormatException("ROI was set wrong");
        }
    }

    /**
     * 
     * @return int array with the roi informations (x,y,width,height)
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if communication
     * with micro manager went wrong
     */
    int[] getRoi() throws CameraException {
        int rawRoi[] = cp.getRoi();
        int len = rawRoi.length;
        int[] extendedRoi = new int[len + 1];
        for (int i = 0; i < len; i++) {
            extendedRoi[i] = rawRoi[i];
        }
        extendedRoi[len] = sendImageSize;
        return extendedRoi;
    }

    /**
     * Sets the exposure time in milliseconds
     * @param time exposure time in milliseconds
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if communication
     * with micro manager went wrong
     */
    void setExposure(double time) throws CameraException {
        cp.setExposure(time);
    }

    /**
     * 
     * @return the exposure time in milliseconds
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if communication
     * with micro manager went wrong
     */
    double getExposure() throws CameraException {
        return cp.getExposure();
    }

    /**
     * Sets a specified configuration
     * @param groupId id of the configuration group
     * @param configId id of the configuration
     * @throws org.fairsim.cameraplugin.CameraPlugin.CameraException if communication
     * with micro manager went wrong
     */
    void setConfig(int groupId, int configId) throws CameraException {
        cp.setConfig(groups[groupId].getNmae(), groups[groupId].getConfig(configId));
    }

    /**
     * 
     * @return groups of the camera
     */
    CameraGroup[] getGroups() {
        return groups;
    }

    /**
     * starts acquisition
     */
    void startAcquisition() {
        gui.startButton.setEnabled(false);
        gui.stopButton.setEnabled(true);
        acquisitionThread = new AcquisitionThread();
        acquisitionThread.start();
    }
    
    /**
     * stops acquisition
     */
    void stopAcquisition() {
        gui.startButton.setEnabled(true);
        gui.stopButton.setEnabled(false);
        if (acquisitionThread != null) {
            acquisitionThread.interrupt();
        }
    }
    
    /**
     * starts acquisition, called over network
     */
    void startNetworkAcquisition() {
        gui.startButton.setEnabled(false);
        acquisitionThread = new AcquisitionThread();
        acquisitionThread.start();
    }

    /**
     * stops the acquisition thread
     */
    void close() {
        if (acquisitionThread != null) {
            acquisitionThread.interrupt();
        }
    }

    /**
     * Thread to acquire images and send them over network
     */
    private class AcquisitionThread extends Thread {

        boolean acquisition;
        boolean imagesQueued;
        boolean imagesSended;
        
        /**
         * offers an image to send it over network
         * @param imgData the image to be send
         * @param count frame number of the image
         * @param timeStamp timestamp of the image
         */
        private void queueImage(short[] imgData, int count, long timeStamp) {
            ImageWrapper iw;
            //System.out.println(mirrored + "/" + sendPixelSize);
            if (mirrored) {
                iw = ImageWrapper.copyImageCropMirrorXCentered(imgData, sendImageSize, sendImageSize, imageWidth, imageHeight, 0, 0, 0, channel, count);
                
            } else {
                iw = ImageWrapper.copyImageCropCentered(imgData, sendImageSize, sendImageSize, imageWidth, imageHeight, 0, 0, 0, channel, count);
            }
            iw.setTimeCamera(timeStamp);
            iw.setTimeCapture(System.currentTimeMillis() * 1000);
            imagesQueued = imagesQueued && isend.queueImage(iw);
            imagesSended = imagesSended && isend.canSend();
        }

        /**
         * acquires images and sends them over network
         */
        public void run() {
            acquisition = imagesQueued = imagesSended = true;
            isend.clearBuffer();
            try {
                // connects image senders
                for (String ip : sendIps) {
                    isend.connect(ip, null);
                }
                
                // starting acquisition
                cp.startSequenceAcquisition();
                int count = 0;
                Tool.Timer t1 = Tool.getTimer();
                t1.start();
                while (acquisition) {
                    if (!cp.isSequenceRunning()) {
                        cp.startSequenceAcquisition();
                    }
                    if (cp.getRemainingImageCount() > 0) {
                        // retrieve image from camera
                        short[] imgData;
                        imgData = (short[]) cp.getNextImage();
                        count++;
                        long timeStamp = Tool.decodeBcdTimestamp(imgData);
                        // send image to reconstruction / capture
                        queueImage(imgData, count, timeStamp);
                        // display image all 59 images & updates queuing/sending color & fps
                        if (count % FPSCOUNTS == 0) {
                            t1.stop();
                            fps = ((FPSCOUNTS * 1000) / t1.msElapsed());
                            t1.start();
                            gui.setFps(fps);
                            if (imagesQueued) gui.setQueuingColor(Color.GREEN);
                            else gui.setQueuingColor(Color.RED);
                            if (imagesSended) gui.setSendingColor(Color.GREEN);
                            else gui.setSendingColor(Color.RED);
                            queued = imagesQueued;
                            sended = imagesSended;
                            imagesQueued = imagesSended = true;
                            gui.view.newImage(0, imgData);
                            gui.view.refresh();
                        }
                    } else {
                        cp.sleepCam(2);
                    }
                    if (isInterrupted()) {
                        acquisition = false;
                        for (String ip : sendIps) {
                            isend.shutdownThreads();
                        }
                    }
                }
                // stops and resets acqusition
                t1.stop();
                cp.stopSequenceAcquisition();
                gui.resetFps();
                gui.resetQueuingColor();
                gui.resetSendingColor();
            } catch (UnknownHostException | CameraException ex) {
                acquisition = false;
                System.err.println(ex);
                gui.showText("AcquisitionThread: " + ex.toString());
                gui.closeWholePlugin();
            }
        }
    }
}
