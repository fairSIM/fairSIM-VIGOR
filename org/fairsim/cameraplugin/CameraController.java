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
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.DataFormatException;
import org.fairsim.cameraplugin.CameraPlugin.CameraException;
import org.fairsim.transport.ImageSender;
import org.fairsim.transport.ImageWrapper;
import org.fairsim.utils.Conf;
import org.fairsim.utils.Tool;

/**
 *
 * @author Mario
 */
public class CameraController {

    private final CameraPlugin cp;
    private final CameraServerGui gui;
    private AcquisitionThread acquisitionThread;
    private int channel;
    private final List<String> sendIps;
    private final CameraGroup[] groups;
    private static final int CAMBUFFER = 1000;
    private static final int FPSCOUNTS = 1000;
    private int[] roi;
    private int imageWidth;
    private int imageHeight;
    private int sendWidth;
    private int sendHeight;
    private boolean mirrored;
    private boolean cropped;
    private final ImageSender isend;
    double  fps;
    boolean queued;
    boolean sended;

    CameraController(CameraPlugin cp) throws IOException, CameraException {
        this.cp = cp;
        isend = new ImageSender();
        sendIps = new ArrayList<>();

        readinFromXML();

        cp.setBuffer(CAMBUFFER);

        String[] grps = cp.getConfigGroups();
        groups = new CameraGroup[grps.length];
        for (int i = 0; i < grps.length; i++) {
            groups[i] = new CameraGroup(grps[i], cp.getConfigs(grps[i]));
        }

        this.gui = new CameraServerGui(imageWidth, imageHeight, this);
    }
    /*
    private void readinFromTXT() throws IOException, CameraException {
        String filename = Tool.getFile(System.getProperty("user.home") + "/documents/fair-sim-ips.txt").getAbsolutePath();
        BufferedReader br = new BufferedReader(new FileReader(filename));
        channel = Integer.parseInt(br.readLine());
        String line = br.readLine();
        String[] flags = line.split(" ");
        mirrored = false;
        cropped = false;
        for (String flag : flags) {
            if (flag.equals("m")) {
                mirrored = true;
            } else if (flag.equals("c")) {
                cropped = true;
            }
        }
        line = br.readLine();
        String[] roiStrings = line.split(" ");
        int x = Integer.parseInt(roiStrings[0]);
        int y = Integer.parseInt(roiStrings[1]);
        int w = Integer.parseInt(roiStrings[2]);
        int h = Integer.parseInt(roiStrings[3]);
        try {
            setRoi(x, y, w, h, true);
        } catch (DataFormatException ex) {
        }
        while ((line = br.readLine()) != null) {
            sendIps.add(line);
        }
        br.close();
    }
    */
    
    private void readinFromXML() throws IOException, CameraException {
        String filename = Tool.getFile(System.getProperty("user.home") + "/documents/fastsim-camera.xml").getAbsolutePath();

        try {
            Conf.Folder cfg = Conf.loadFile(filename).r().cd("camera-settings");
            channel = cfg.getInt("Channel").val();
            String[] flags = cfg.getStr("Flags").val().split(" ");
            int[] roi = cfg.getInt("Roi").vals();
            String[] ips = cfg.getStr("SendIps").val().split(" ");
            mirrored = false;
            cropped = false;
            for (String flag : flags) {
                if (flag.equals("mirrored")) {
                    mirrored = true;
                } else if (flag.equals("cropped")) {
                    cropped = true;
                }
            }
            try {
                setRoi(roi[0], roi[1], roi[2], roi[3], true);
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

    }

    void setRoi(int x, int y, int width, int height) throws CameraException, DataFormatException {
        setRoi(x, y, width, height, false);
    }

    private void setRoi(int x, int y, int width, int height, boolean firstTime) throws CameraException, DataFormatException {
        cp.stopSequenceAcquisition();
        cp.setROI(x, y, width, height);
        roi = cp.getROI();
        imageWidth = cp.getImageWidth();
        imageHeight = cp.getImageHeight();
        if (imageWidth != roi[2] || imageHeight != roi[3]) {
            throw new RuntimeException("This should never happen!");
        }
        if (imageWidth < 512 || imageHeight < 512) {
            sendWidth = sendHeight = 256;
        } else {
            sendWidth = sendHeight = 512;
        }
        if (!firstTime) {
            gui.refreshView(imageWidth, imageHeight);
        }
        if (roi[0] != x || roi[1] != y || roi[2] != width || roi[3] != height) {
            throw new DataFormatException("ROI was set wrong");
        }
    }

    int[] getRoi() throws CameraException {
        return cp.getROI();
    }

    void setExposure(double time) throws CameraException {
        cp.setExposure(time);
    }

    double getExposure() throws CameraException {
        return cp.getExposure();
    }

    void setConfig(int groupId, int configId) throws CameraException {
        cp.setConfig(groups[groupId].getNmae(), groups[groupId].getConfig(configId));
    }

    CameraGroup[] getGroups() {
        return groups;
    }

    void startAcquisition() {
        gui.startButton.setEnabled(false);
        gui.stopButton.setEnabled(true);
        acquisitionThread = new AcquisitionThread();
        acquisitionThread.start();
    }

    void stopAcquisition() {
        gui.startButton.setEnabled(true);
        gui.stopButton.setEnabled(false);
        if (acquisitionThread != null) {
            acquisitionThread.interrupt();
        }
    }

    void close() {
        if (acquisitionThread != null) {
            acquisitionThread.interrupt();
        }
    }

    private class AcquisitionThread extends Thread {

        boolean acquisition;
        boolean imagesQueued;
        boolean imagesSended;

        private void queueImage(short[] imgData, int count, long timeStamp) {
            ImageWrapper iw;
            if (cropped && !mirrored) {
                iw = ImageWrapper.copyImageCrop(imgData, sendWidth, sendHeight, imageWidth, imageHeight, 0, 0, 0, channel, count);
            } else if (mirrored && !cropped) {
                iw = ImageWrapper.copyImageMirrorX(imgData, imageWidth, imageHeight, 0, 0, 0, channel, count);
            } else if (mirrored && cropped) {
                iw = ImageWrapper.copyImageCropMirrorX(imgData, sendWidth, sendHeight, imageWidth, imageHeight, 0, 0, 0, channel, count);
            } else {
                iw = ImageWrapper.copyImage(imgData, imageWidth, imageHeight, 0, 0, 0, channel, count);
            }
            iw.setTimeCamera(timeStamp);
            iw.setTimeCapture(System.currentTimeMillis() * 1000);
            imagesQueued = imagesQueued && isend.queueImage(iw);
            imagesSended = imagesSended && isend.canSend();
        }

        public void run() {
            acquisition = imagesQueued = imagesSended = true;
            isend.clearBuffer();
            try {
                for (String ip : sendIps) {
                    isend.connect(ip, null);
                }
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
                        // sets framerate all 1000 frames
                        if (count % FPSCOUNTS == 0) {
                            t1.stop();
                            fps = ((FPSCOUNTS * 1000) / t1.msElapsed());
                            t1.start();
                        }
                        // display image all 59 images & updates queuing/sending color
                        if (count % 59 == 0) {
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
                t1.stop();
                cp.stopSequenceAcquisition();
                gui.resetQueuingColor();
                gui.resetSendingColor();
            } catch (UnknownHostException | CameraException ex) {
                acquisition = false;
                gui.showText("AcquisitionThread: " + ex.toString());
                gui.closeWholePlugin();
            }
        }
    }
}
