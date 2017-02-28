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

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import mmcorej.CMMCore;
import org.fairsim.sim_gui.PlainImageDisplay;
import org.fairsim.transport.ImageSender;
import org.fairsim.transport.ImageWrapper;
import org.fairsim.utils.Tool;
import org.micromanager.api.ScriptInterface;

/**
 *
 * @author m.lachetta
 */
public class CameraPlugin implements org.micromanager.api.MMPlugin {

    public static String menuName = "Fast SIM Camera Controller";
    public static String tooltipDescription = "Micro manager plugin to controll the cameras of the fast SIM setup";
    private static final int ROILENGTH = 4;
    ScriptInterface si;
    CMMCore mmc;
    CameraController cc;
    /*
    int channel;
    List<String> sendIps;
    CameraGroup[] groups;
    ImageWrapper iw;
    
    int roiX;
    int roiY;
    int roiWidth;
    int roiHeight;
    int imageWidth;
    int imageHeight;
    int sendWidth;
    int sendHeight;
    boolean mirrored;
    boolean croped;
    ImageSender isend;
    PlainImageDisplay view;
    boolean imageQueued;
    String fps;
    boolean acquisition;
    AcquisitionThread acquisitionThread;
    CameraServerGui guiFrame;

    void initPlugin(ScriptInterface gui) throws IOException {
        //this.gui = gui;
        this.mmc = gui.getMMCore();
        this.channel = -1;
        sendIps = new ArrayList<>();
        isend = new ImageSender();
        imageQueued = false;
        String fps = "";
        acquisition = false;
        //acquisitionThread = new AcquisitionThread();
        readinInfo();
    }

    void prepareAcquisition(int xROI, int yROI, int wROI, int hROI) throws CameraException, UnknownHostException {
        try {
            stopSequenceAcquisition();
            setRoi(xROI, yROI, wROI, hROI);
            this.setBuffer();
            guiFrame = new CameraServerGui(imageWidth, imageHeight, this);
            guiFrame.setVisible(true);
            view = guiFrame.view;
            connect();
        } catch (CameraException ex) {
            throw ex;
        }
    }

    void startAcquisition() {
        guiFrame.startButton.setEnabled(false);
        guiFrame.stopButton.setEnabled(true);
        acquisitionThread = new AcquisitionThread();
        acquisitionThread.start();
    }

    void stopAcquisition() {
        guiFrame.startButton.setEnabled(true);
        guiFrame.stopButton.setEnabled(false);
        acquisition = false;
        if (acquisitionThread != null) {
            acquisitionThread.interrupt();
        }
    }

    void shutdownThreads() {
        acquisition = false;
        if (acquisitionThread != null) {
            acquisitionThread.interrupt();
        }
        isend.shutdownThreads();
    }

    private class AcquisitionThread extends Thread {

        public void run() {
            acquisition = true;
            try {
                startSequenceAcquisition();
                int count = 0;
                Tool.Timer t1 = Tool.getTimer();
                t1.start();
                while (acquisition) {
                    if (!isSequenceRunning()) {
                        startSequenceAcquisition();
                    }
                    if (getRemainingImageCount() > 0) {
                        // retrieve image from camera
                        short[] imgData;
                        imgData = (short[]) getNextImage();
                        count++;
                        long timeStamp = Tool.decodeBcdTimestamp(imgData);
                        // send image to reconstruction / capture
                        queueImage(imgData, count, timeStamp);
                        // sets framerate all 1000 frames
                        if (count % 1000 == 0) {
                            t1.stop();
                            fps = ((1000 * 1000) / t1.msElapsed()) + " "
                                    + Tool.readableTimeStampMillis(timeStamp / 1000, true) + " ."
                                    + timeStamp % 1000000 + " us";
                            t1.start();
                        }
                        // display image all 59 images
                        if (count % 59 == 0) {
                            view.newImage(0, imgData);
                            view.refresh();
                        }
                    } else {
                        sleepCam(2);
                    }
                }
                t1.stop();
                stopSequenceAcquisition();
            } catch (CameraException ex) {
                acquisition = false;
                displayMessage("AcquisitionThread: " + ex.toString());
            }
        }
    }

    private void displayMessage(String message) {
        guiFrame.showText(message);
    }

    private void queueImage(short[] imgData, int count, long timeStamp) {
        if (croped && !mirrored) {
            iw = ImageWrapper.copyImageCrop(imgData, sendWidth, sendHeight, imageWidth, imageHeight, 0, 0, 0, channel, count);
        } else if (mirrored && !croped) {
            iw = ImageWrapper.copyImageMirrorX(imgData, imageWidth, imageHeight, 0, 0, 0, channel, count);
        } else if (mirrored && croped) {
            iw = ImageWrapper.copyImageCropMirrorX(imgData, sendWidth, sendHeight, imageWidth, imageHeight, 0, 0, 0, channel, count);
        } else {
            iw = ImageWrapper.copyImage(imgData, imageWidth, imageHeight, 0, 0, 0, channel, count);
        }
        iw.setTimeCamera(timeStamp);
        iw.setTimeCapture(System.currentTimeMillis() * 1000);
        imageQueued = isend.queueImage(iw);
    }

    private void connect() throws UnknownHostException {
        for (String ip : sendIps) {
            isend.connect(ip, null);
        }
    }

    void setRoi(int x, int y, int width, int height) throws CameraException {
        setROI(x, y, width, height);
        updateRoi();
        if (roiX != x || roiY != y || roiWidth != width || roiHeight != height) {
            System.out.println("ROI was set wrong \n"
                    + "x = " + roiX + "; y = " + roiY + "; width = " + roiWidth + "; height = " + roiHeight);
        }
        if (roiWidth < 512 || roiHeight < 512) {
            sendWidth = sendHeight = 256;
        } else {
            sendWidth = sendHeight = 512;
        }
        imageWidth = getImageWidth();
        imageHeight = getImageHeight();
        if (guiFrame != null) {
            guiFrame.refreshView(imageWidth, imageHeight);
            view = guiFrame.view;
        }
    }

    private void updateRoi() throws CameraException {
        int[] roi = getROI();
        roiX = roi[0];
        roiY = roi[1];
        roiWidth = roi[2];
        roiHeight = roi[3];
    }

    void setConfig(int groupId, int configId) throws CameraException {
        setConfig(groups[groupId].getNmae(), groups[groupId].getConfig(configId));
    }

    private void updateGroups() throws CameraException {
        String[] grps = getConfigGroups();
        groups = new CameraGroup[grps.length];
        for (int i = 0; i < grps.length; i++) {
            groups[i] = new CameraGroup(grps[i], getConfigs(grps[i]));
        }
    }

    CameraGroup[] getGroups() throws CameraException {
        updateGroups();
        return groups;
    }

    private void readinInfo() throws IOException {
        String filename = "";
        try {
            filename = Tool.getFile(System.getProperty("user.home") + "/documents/fair-sim-ips.txt").getAbsolutePath();
            BufferedReader br = new BufferedReader(new FileReader(filename));
            channel = Integer.parseInt(br.readLine());
            String line = br.readLine();
            mirrored = false;
            croped = false;
            if (line.equals("m")) {
                mirrored = true;
            } else if (line.equals("c")) {
                croped = true;
            } else {
                throw new IOException();
            }
            while ((line = br.readLine()) != null) {
                sendIps.add(line);
            }
        } catch (IOException ex) {
            throw new IOException("Readin failed: '" + filename + "'");
        }
    }
    */
    //wrapped methods:
    
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
        try {
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
    
    public int[] getROI() throws CameraException {
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

        CameraException(String massage) {
            super("Error: " + massage);
            //shutdownThreads();
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
        this.si = si;
        mmc = si.getMMCore();
        try {
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
