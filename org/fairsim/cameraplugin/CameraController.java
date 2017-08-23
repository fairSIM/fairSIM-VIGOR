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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
    private final int[] channels;
    private final List<String> sendIps;
    private final CameraGroup[] groups;
    private static final int CAMBUFFER = 512;
    private static final int FPSCOUNTS = 59;
    private int[] roi;
    private final int[] bigRoi, smallRoi;
    private int imageWidth;
    private int imageHeight;
    private int sendImageSize;
    private final boolean[] mirrored;
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
        String filename = Tool.getFile(System.getProperty("user.home") + "/documents/fairsim-camera.xml").getAbsolutePath();
        try {
            Conf.Folder cfg = Conf.loadFile(filename).r().cd("camera-settings");
            channels = cfg.getInt("Channel").vals();
            if (channels.length < cp.getChannelCount()) throw new IOException(cp.getChannelCount() +
                    " cams in use, but only " + channels.length + " in config file " + filename);
            String[] mx = cfg.getStr("MirroredX").val().split(" ");
            if (mx.length != channels.length) throw new IOException("Amount of MirroredX flags unequal to amount of channels");
            mirrored = new boolean[mx.length];
            for (int i = 0; i < mx.length; i++) {
                if (mx[i].equals("true")) mirrored[i] = true;
                else if (mx[i].equals("false")) mirrored[i] = false;
                else throw new IOException("Only true or false as MirroredX flags allowed");
            }
            bigRoi = cfg.getInt("BigRoi").vals();
            smallRoi = cfg.getInt("SmallRoi").vals();
            String[] ips = cfg.getStr("SendIps").val().split(" ");
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
            ex.printStackTrace();
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
        this.gui = new CameraServerGui(sendImageSize, sendImageSize, this);
    }
    
    int[] getChannels() {
        return channels;
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
            gui.refreshView(sendImageSize, sendImageSize);
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
        if (groupId >= 0 && configId >= 0) cp.setConfig(groups[groupId].getNmae(), groups[groupId].getConfig(configId));
        else throw new IllegalArgumentException("Group or config id < 0");
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
        try {
            if (cp.isSequenceRunning()) return;
        } catch (CameraException ex) {
            throw new RuntimeException(ex);
        }
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
        private final Map<Integer, Long> seqNrMapping = new HashMap<>();
        
        /**
         * offers an image to send it over network
         * @param imgData the image to be send
         * @param count frame number of the image
         * @param timeStamp timestamp of the image
         */
        private short[] queueImage(int channelIdx, short[] imgData, int count, long timeStamp) {
            ImageWrapper iw;            
            if (mirrored[channelIdx]) {
                iw = ImageWrapper.copyImageCropMirrorXCentered(imgData, sendImageSize, sendImageSize, imageWidth, imageHeight, 0, 0, 0, channels[channelIdx], count);
                
            } else {
                iw = ImageWrapper.copyImageCropCentered(imgData, sendImageSize, sendImageSize, imageWidth, imageHeight, 0, 0, 0, channels[channelIdx], count);
            }
            iw.setTimeCamera(timeStamp);
            iw.setTimeCapture(System.currentTimeMillis() * 1000);
            
            // set seqNr
            Long sn = seqNrMapping.get(channelIdx);
            if (sn == null) {
                sn = (long) (Math.random() * Math.pow(2, 30)) << 32;
                seqNrMapping.put(channelIdx, sn);
            }
            iw.setSeqNr(sn);
            seqNrMapping.put(channelIdx, sn + 1);
            
            imagesQueued = imagesQueued && isend.queueImage(iw);
            imagesSended = imagesSended && isend.canSend();
            
            return iw.getPixels();
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
                //cp.startSequenceAcquisition();
                int count = 0;
                Tool.Timer t1 = Tool.getTimer();
                t1.start();
                while (acquisition) {
                    if (!cp.isSequenceRunning()) {
                        cp.startSequenceAcquisition();
                    }
                    if (cp.getRemainingImageCount() > 0) {
                        // retrieve image from camera
                        CameraPlugin.ChanneldImage chImg = cp.getNextImage();
                        int chIdx = chImg.chIdx;
                        short[] imgData = chImg.img;
                        count++;
                        long timeStamp = Tool.decodeBcdTimestamp(imgData);
                        // send image to reconstruction / capture
                        short[] sendData = queueImage(chIdx, imgData, count, timeStamp);
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
                            gui.view.newImage(chIdx, sendData);
                            gui.view.refresh();
                        }
                    } else {
                        cp.sleep(2);
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
                System.err.println("AcquisitionThread: " + ex);
                gui.showText("AcquisitionThread: " + ex.toString());
                ex.printStackTrace();
                gui.closeWholePlugin();
            }
        }
    }
}
