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
package org.fairsim.livemode;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;

import java.util.Map;
import java.util.TreeMap;
import java.awt.Color;
import java.nio.BufferOverflowException;

import org.fairsim.transport.ImageReceiver;
import org.fairsim.transport.ImageWrapper;
import org.fairsim.utils.Tool;
import org.fairsim.linalg.MTool;

import org.fairsim.utils.Conf;

/**
 * Takes raw images from the network listener, syncs to SIM sequence, passes
 * them to ReconstructionRunner
 */
public class SimSequenceExtractor {

    final ImageReceiver imgRecv;
    final ReconstructionRunner reconRunner;
    final int nrChannels, rawsPerRecon;

    int seqCount;
    int syncFrameAvr;
    int syncFrameDelay;

    final PerChannelBuffer[] channels;
    private Map< Integer, PerChannelBuffer> channelMapping;
    final private LiveControlPanel livePanel;
    
    private boolean pause;

    /**
     * Links ir to rr
     */
    public SimSequenceExtractor(Conf.Folder cfg,
            ImageReceiver ir, ReconstructionRunner rr, LiveControlPanel ll)
            throws Conf.EntryNotFoundException {
        this.imgRecv = ir;
        this.reconRunner = rr;
        this.nrChannels = rr.nrChannels;
        this.livePanel = ll;

        this.seqCount = cfg.getInt("SyncFrameFreq").val();
        this.syncFrameAvr = cfg.getInt("SyncFrameAvr").val();
        this.syncFrameDelay = cfg.getInt("SyncFrameDelay").val();
        
        rawsPerRecon = reconRunner.nrPhases * reconRunner.nrDirs;
        pause = false;

        channelMapping = new TreeMap<Integer, PerChannelBuffer>();
        channels = new PerChannelBuffer[nrChannels];
        for (int i = 0; i < nrChannels; i++) {
            channels[i] = new PerChannelBuffer(rawsPerRecon * rawsPerRecon * 10,
                    reconRunner.getChannel(i).chNumber, i, cfg.getInt("SyncMode").val());
            Tool.trace("Created receive buffer for channel: "
                    + reconRunner.getChannel(i).chNumber);
            channelMapping.put(reconRunner.getChannel(i).chNumber, channels[i]);
        }

        // start an ImageSorter
        ImageSorter is = new ImageSorter();
        is.start();

        // start the per-channel sequence detection
        for (PerChannelBuffer pcb : channels) {
            pcb.start();
        }

        JoinedChannelBuffer jcb = new JoinedChannelBuffer();
        jcb.start();

        Tool.trace("Image sequency detection started");
    }

    /**
     * Clears all receive and sort buffers for resyncing to the stream
     */
    public void resetChannelBufferThreads() {
        for (int i = 0; i < nrChannels; i++) {
            channels[i].restartThread = true;
            channels[i].interrupt();
        }
    }

    public void clearBuffers() {
        for (PerChannelBuffer i : channels) {
            i.clearBuffers();
        }
    }

    public int getSyncDelay() {
        return syncFrameDelay;
    }

    public int getSyncAvr() {
        return syncFrameAvr;
    }

    public int getSyncFreq() {
        return seqCount;
    }

    public void setSyncDelay(int delay) {
        if (delay > 0) {
            syncFrameDelay = delay;
        } else {
            throw new NumberFormatException("syncFrameDelay has to be positive");
        }
    }

    public void setSyncAvr(int avr) {
        if (syncFrameAvr > 0) {
            syncFrameAvr = avr;
        } else {
            throw new NumberFormatException("syncFrameAvr has to be positive");
        }
    }

    public void setSyncFreq(int freq) {
        if (seqCount > 0) {
            seqCount = freq;
        } else {
            throw new NumberFormatException("syncFreq has to be positive");
        }
    }

    public void pause(boolean b) {
        pause = b;
    }

    /**
     * Take images for the gereral queue, sort them by channel
     */
    class ImageSorter extends Thread {
        
        @Override
        public void run() {

            while (true) {
                if (!pause) {
                    ImageWrapper iw = imgRecv.takeImage();
                    int chNr = iw.pos1();	// pos1 holds the data packets image channel
                    PerChannelBuffer pc = channelMapping.get(chNr);
                    if (pc == null) {
                        Tool.error("ImgSort: received data packet w/o channel: " + chNr, false);
                    } else {
                        pc.pushImg(iw);
                    }
                } else {
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException ex) {
                    }
                }
            }
        }
    }

    /**
     * Takes tuples, triples, etc of SIM images, and forwards them for
     * reconstruction
     */
    class JoinedChannelBuffer extends Thread {

        public void run() {

            while (true) {

                short[][][] res = new short[nrChannels][][];

                try {
                    for (int c = 0; c < nrChannels; c++) {
                        res[c] = channels[c].simSeq.take();
                    }
                } catch (InterruptedException e) {
                    Tool.trace("Channel joiner interrupted, why?");
                    continue;
                }
                reconRunner.queueImage(res);
            }
        }

    }

    /**
     * Sorts through the raw images, waiting for a sync frame, assembles SIM
     * sequences
     */
    class PerChannelBuffer extends Thread {

        BlockingQueue<ImageWrapper> rawImgs;
        BlockingQueue<short[][]> simSeq;
        final SortBuffer sortBuffer;
        
        final int queueSize, chNumber, chIndex;
        boolean restartThread;

        int missedRaw = 0;
        int missedSim = 0;
        int noSyncSince = 0;
        long syncFrameCount = 0;
        long seqNr;
        final int syncMode;
        static final int SYNCJITTER = 50;

        int queryRaw() {
            return rawImgs.size();
        }

        int querySim() {
            return simSeq.size();
        }

        PerChannelBuffer(int queueSize, int chNumber, int chIndex, int syncMode) {
            this.queueSize = queueSize;
            this.chNumber = chNumber;
            this.chIndex = chIndex;
            this.syncMode = syncMode;
            rawImgs = new ArrayBlockingQueue<>(queueSize);
            simSeq = new ArrayBlockingQueue<>(queueSize);
            sortBuffer = new SortBuffer(20);
            restartThread = false;
        }
        
        /**
         * forwards an image to this
         * @param iw image
         */
        void pushImg(ImageWrapper iw) {
            boolean ok = rawImgs.offer(iw);
            if (!ok) {
                missedRaw++;
                Tool.error("missing raw frame at PerChannelBuffer.pushImg", false);
            }
        }
        
        /**
         * 
         * @return the next raw frame from the raw queue or the sort buffer
         * including sync frames
         * @throws InterruptedException if gets an interrupt
         */
        private ImageWrapper getSorted() throws InterruptedException {
            ImageWrapper image = null;
            if (sortBuffer.isEmpty()) {
                return sortFromQueue();
            } else {
                image = sortBuffer.getNext();
                if (image != null) {
                    seqNr++;
                    return image;
                } else {
                    return sortFromQueue();
                }
            }
        }
        
        /**
         * 
         * @return the next image from the raw queue
         * @throws InterruptedException 
         */
        private ImageWrapper sortFromQueue() throws InterruptedException {
            ImageWrapper image = rawImgs.take();
            if (image.seqNr() == seqNr) {
                seqNr++;
                return image;
            } else {
                try {
                    sortBuffer.add(image);
                } catch (BufferOverflowException ex) {
                    seqNr++;
                    Tool.error("missing raw frame at PerChannelBuffer.getSorted " + sortBuffer.buffer.size() + " " + image.pos2(), false);
                    setSeqNr();
                }
                return getSorted();
            }
        }
        
        /**
         * clearing all buffers of this except the sort buffer
         */
        void clearBuffers() {
            rawImgs.clear();
            simSeq.clear();
            syncFrameCount = 0;
        }
        
        /**
         * fills the sort buffer and sets the minimum of its containing seqNr
         * as new starting seqNr for this
         */
        void setSeqNr() {
            sortBuffer.buffer.clear();
            seqNr = Long.MAX_VALUE;
            while (sortBuffer.buffer.size() < rawsPerRecon) {
                try {
                    ImageWrapper iw = rawImgs.take();
                    sortBuffer.add(iw);
                    if (iw.seqNr() < seqNr) {
                        seqNr = iw.seqNr();
                    }
                } catch (InterruptedException ex) {
                    if (!restartThread) {
                        Tool.error("Image sorting thread [" + this.chIndex + "] interupted, why?", false);
                    } else {
                        restartThread = false;
                        continue;
                    }
                }
                if (seqNr == Long.MAX_VALUE) {
                    throw new RuntimeException("seqNr was not set");
                }
            }
        }
        
        private boolean isTimeDelaySync(long curTimeStamp, long lastTimeStamp) {
            return curTimeStamp - lastTimeStamp  > syncFrameDelay;
        }
        
        /**
         * 
         * @param curTimeStamp current timestamp
         * @param lastTimeStamp last timestamp before the current timestamp
         * @return true if two timestamps are sync timestamps
         */
        private boolean isTimeStampSync(long curTimeStamp, long lastTimeStamp) {
            return Math.abs(curTimeStamp - lastTimeStamp - syncFrameDelay) < SYNCJITTER;
        }
        
        /**
         * 
         * @param pxl image
         * @return true if image is detected as an average based sync frame
         */
        private boolean isAvrSync(short[] pxl) {
            return MTool.avr_ushort(pxl) > syncFrameAvr;
        }

        /**
         * Sequence detection, emptying rawImgs, filling simSeq
         */
        @Override
        public void run() {

            final int nrRawPerSeq = reconRunner.nrDirs * reconRunner.nrPhases;
            setSeqNr();
            long lastTimeStamp = 0;
            long curTimeStamp = 0;
            while (true) {

                try {
                    int counter = 0;
                    // first, loop over incoming images until we find a sync frame
                    while (true) {

                        // take a frame, get it's timestamp
                        ImageWrapper iwSync = getSorted();
                        curTimeStamp = iwSync.timeCamera();
                        if (syncMode == 0) {
                            // version 0 (for camera with time-stamp, like IDS µEye)
                            if (isTimeDelaySync(curTimeStamp, lastTimeStamp)) {
                                sortBuffer.add(iwSync);
                                seqNr--;
                                syncFrameCount++;
                                long count = syncFrameCount / 5;
                                Color bg = (count % 2 == 0) ? (Color.BLACK) : (Color.GREEN);
                                livePanel.syncButtons[chIndex].setBackground(bg);
                                break;
                            }
                        } else if (syncMode == 1) {
                            // version 1 (for camera with precise time-stamp, like PCO)
                            if (isTimeStampSync(curTimeStamp, lastTimeStamp)) {
                                //Tool.tell("SYNC "+chNumber+": via timestamp/PCO");
                                syncFrameCount++;
                                long count = syncFrameCount / 5;
                                Color bg = (count % 2 == 0) ? (Color.BLACK) : (Color.GREEN);
                                livePanel.syncButtons[chIndex].setBackground(bg);
                                break;
                            }

                            

                            // version 2 (for camera w/o timestamp, bright LED):
                            short pxl[] = iwSync.getPixels();
                            if (isAvrSync(pxl)) {
                                syncFrameCount++;
                                long count = syncFrameCount / 5;
                                Color bg = (count % 2 == 0) ? (Color.BLACK) : (Color.GREEN);
                                livePanel.syncButtons[chIndex].setBackground(bg);
                                //Tool.tell("SYNC "+chNumber+": via bright frame");
                                getSorted(); // ignore the next frame
                                break;
                            }
                        } else {
                            String error = "missmatch in syncMode, expected 0 or 1, found " + syncMode;
                            Tool.error(error);
                            throw new RuntimeException(error);
                        }
                        
                        lastTimeStamp = curTimeStamp;
                        counter++;
                        
                        if (counter >= nrRawPerSeq * seqCount) {
                            Tool.trace("No sync frame found " + iwSync.seqNr() + "/" + nrRawPerSeq + "/" + seqCount);
                            counter = 0;
                        }
                    }

                    // then, copy the next n x m frames
                    for (int k = 0; k < seqCount; k++) {

                        short[][] simPxls = new short[nrRawPerSeq][];

                        for (int i = 0; i < nrRawPerSeq; i++) {
                            ImageWrapper iw = getSorted();
                            simPxls[i] = iw.getPixels();
                            if(syncMode == 0) {
                                curTimeStamp = iw.timeCamera();
                                if (isTimeDelaySync(curTimeStamp, lastTimeStamp) && !(k == 0 && i == 0))
                                    Tool.trace("Mode 0 timeStamp sync frame found in sequence " + iw.seqNr() + " " + (curTimeStamp - lastTimeStamp) + " " + iw.pos1());
                                lastTimeStamp = curTimeStamp;
                            } else if (syncMode == 1) {
                                curTimeStamp = iw.timeCamera();
                                if (isTimeStampSync(curTimeStamp, lastTimeStamp))
                                    Tool.trace("Mode 1 timeStamp sync frame found in sequence " + iw.seqNr());
                                lastTimeStamp = curTimeStamp;
                                if (isAvrSync(simPxls[i]))
                                    Tool.trace("Mode 1 avr sync frame found in sequence " + iw.seqNr());
                            }
                        }

                        boolean ok = simSeq.offer(simPxls);
                        if (!ok) {
                            missedSim++;
                        }
                    }

                } catch (InterruptedException e) {
                    if (!restartThread) {
                        Tool.error("Image sorting thread [" + this.chIndex + "] interupted, why?", false);
                    } else {
                        restartThread = false;
                        setSeqNr();
                        continue;
                    }
                }
            }
        }
        
        /**
         * class that provides a buffer for sorting images on the basis of seqNr
         */
        class SortBuffer {

            final int MAXSIZE;
            Map<Long, ImageWrapper> buffer;

            SortBuffer(int maxSize) {
                MAXSIZE = maxSize;
                buffer = new TreeMap<>();
            }
            
            ImageWrapper getNext() {
                return buffer.remove(seqNr);
            }
            
            void add(ImageWrapper image) throws BufferOverflowException{
                if (!isFull()) {
                    ImageWrapper nullCheck = buffer.put(image.seqNr(), image);
                    if (nullCheck != null) {
                        String message = "Same seqNr " + nullCheck.seqNr() + " " + image.seqNr();
                        Tool.error(message, true);
                        throw new RuntimeException(message);
                    }
                }
                else throw new BufferOverflowException();
            }
            
            boolean isEmpty() {
                return buffer.isEmpty();
            }
            
            boolean isFull() {
                return buffer.size() >= MAXSIZE;
            }
            
            int getCapacityPercent() {
                return (buffer.size() * 100 / MAXSIZE);
            }

        }
    }

}
