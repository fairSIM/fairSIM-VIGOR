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

import org.fairsim.sim_algorithm.*;
import org.fairsim.accel.*;
import org.fairsim.linalg.*;
import org.fairsim.utils.Tool;

import org.fairsim.utils.Conf;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.fairsim.registration.Registration;

/**
 * Manages a collection of reconstruction threads and parameter fitting.
 */
public class ReconstructionRunner {

    private final int startingImageSize;
    public int width, height;
    public final int nrChannels;
    public final int nrDirs, nrPhases, nrBands;
    public final int nrThreads;

    private VectorFactory avf;
    private boolean autostart, stopReconThreads;

    private short[][][] latestImage;
    private Vec2d.Real[] latestReconVec;
    public Lock[] latestReconLock;

    private BlockingQueue<short[][][]> imgsToReconstruct;
    private int missedDueToQueueFull;
    final private PerChannel[] channels;

    BlockingQueue<Vec2d.Real[]> finalWidefield;
    BlockingQueue<Vec2d.Real[]> finalRecon;

    BlockingQueue<Integer> doFilterUpdate = new ArrayBlockingQueue<Integer>(16);
    BlockingQueue<Tool.Tuple<Integer, Tool.Callback<SimParam>>> doParameterRefit
            = new ArrayBlockingQueue<Tool.Tuple<Integer, Tool.Callback<SimParam>>>(16);

    private final ReconstructionThread[] reconThreads;
    private FilterUpdateThread fut;
    private ParameterRefitThread prt;

    /**
     * Parameters that have to be set per-channel
     */
    public static class PerChannel {

        int chNumber;
        float offset;
        final double[] startingPx, startingPy;
        SimParam param;
        String label;

        // TODO: setupFilter() reads this, though it could
        // be stored directly with the param.otf()...
        double wienParam = 0.05;
        double attStr = 0.99;
        double attFWHM = 1.25;
        boolean useAttenuation = true;
        
        PerChannel(Conf.Folder fld, int imageSize, double microns, String channel, int nrDirs) throws Conf.EntryNotFoundException {
            param = SimParam.loadConfig(fld);
            param.setPxlSize(imageSize, microns);
            OtfProvider otf = OtfProvider.loadFromConfig(fld);
            param.otf(otf);
            offset = (float) fld.getDbl("offset").val();
            chNumber = fld.getInt("ChannelNumber").val();
            label = channel;
            startingPx = new double[nrDirs];
            startingPy = new double[nrDirs];
            for (int i = 0; i < nrDirs; i++) {
                startingPx[i] = param.dir(i).getPxPy(1)[0];
                startingPy[i] = param.dir(i).getPxPy(1)[1];
            }
        }
    }

    public PerChannel getChannel(int i) {
        return channels[i];
    }

    /**
     * Queue image set for reconstruction
     */
    public boolean queueImage(short[][][] imgs) {
        // offer image to the reconstruction queue
        boolean ok = imgsToReconstruct.offer(imgs);
        if (!ok) {
            missedDueToQueueFull++;
        }

        // save the last image
        latestImage = imgs;
        return ok;
    }

    public ReconstructionRunner(Conf.Folder cfg,
            VectorFactory avf, String[] whichChannels)
            throws Conf.EntryNotFoundException {
        this(cfg, avf, whichChannels, true);
    }

    /**
     * Reads from cfg-folder all channels in 'channels'
     */
    public ReconstructionRunner(Conf.Folder cfg,
            VectorFactory avf, String[] whichChannels, boolean autostart)
            throws Conf.EntryNotFoundException {

        this.avf = avf;
        this.autostart = autostart;

        nrThreads = cfg.getInt("ReconThreads").val();
        imgsToReconstruct = new ArrayBlockingQueue<short[][][]>(maxInReconQueue());

        finalWidefield = new ArrayBlockingQueue<Vec2d.Real[]>(maxInWidefieldQueue());
        finalRecon = new ArrayBlockingQueue<Vec2d.Real[]>(maxInFinalQueue());

        startingImageSize = cfg.getInt("RawPxlCount").val();
        height = width = startingImageSize;
        double microns = cfg.getDbl("RawPxlSize").val();

        nrPhases = cfg.getInt("NrPhases").val();
        nrDirs = cfg.getInt("NrAngles").val();
        nrBands = cfg.getInt("NrBands").val();

        // init per-channel information
        this.nrChannels = whichChannels.length;
        latestReconVec = new Vec2d.Real[nrChannels];
        latestReconLock = new ReentrantLock[nrChannels];
        channels = new PerChannel[nrChannels];
        initLatestReconVec();

        // load initial SIM-param from file
        for (int i = 0; i < nrChannels; i++) {
            Conf.Folder fld = cfg.cd("channel-" + whichChannels[i]);
            channels[i] = new PerChannel(fld, startingImageSize, microns, whichChannels[i], nrDirs);
            /*
            // TODO: this should be moved to a "PerChannel" constructor
            channels[i].param = SimParam.loadConfig(fld);
            channels[i].param.setPxlSize(startingImageSize, microns);
            OtfProvider otf = OtfProvider.loadFromConfig(fld);
            channels[i].param.otf(otf);
            channels[i].offset = (float) fld.getDbl("offset").val();
            channels[i].chNumber = fld.getInt("ChannelNumber").val();
            channels[i].label = whichChannels[i];
            */
        }

        // create and start reconstruction threads
        reconThreads = new ReconstructionThread[nrThreads];
        
        startThreads();
    }
    
    private void startThreads() {
        stopReconThreads = false;
        for (int i = 0; i < nrThreads; i++) {
            reconThreads[i] = new ReconstructionThread(avf);
            if (autostart) {
                reconThreads[i].start();
                Tool.trace("Started recon thread: " + i);
            }
        }

        // precompute filters for all threads
        fut = new FilterUpdateThread();
        for (int ch = 0; ch < nrChannels; ch++) {
            fut.setupFilters(ch);
        }
        
        prt = new ParameterRefitThread();

        if (autostart) {
            fut.start();
            prt.start();
            Tool.trace("Started parameter fit/update thread: ");
        }
    }

    private void stopThreads() {
        stopReconThreads = true;
        if (fut != null) {
            fut.interrupt();
        }
        if (prt != null) {
            prt.interrupt();
        }
        try {
            fut.join();
            prt.join();
            for (ReconstructionThread rt : reconThreads) {
                rt.join();
            }
        } catch (InterruptedException ex) {
            throw new RuntimeException(ex);
        }
        Tool.trace("Stopped ReconstructionThreads.");
    }

    public void setImageSize(int pixelSize) {
        stopThreads();
        this.width = pixelSize;
        this.height = pixelSize;
        imgsToReconstruct.clear();
        initLatestReconVec();
        //double shiftFactor = (double)pixelSize / (double)startingImageSize;
        
        //System.out.println("pixelSize: " + pixelSize + "\t startingImageSize: " + startingImageSize + "\t shiftFactor: " + shiftFactor);
        for (int ch = 0; ch < nrChannels; ch++) {
            channels[ch].param.setPxlSize(pixelSize);
            /*
            for (int i = 0; i < nrDirs; i++) {
                //System.out.println("shift_start: " + channels[ch].startingPx[i] + "/" + channels[ch].startingPy[i]);
                //System.out.println("shift_before: " + channels[ch].param.dir(i).getPxPy(1)[0] + "/" + channels[ch].param.dir(i).getPxPy(1)[1]);
                //channels[ch].param.dir(i).setPxPy(channels[ch].startingPx[i] * shiftFactor * (nrBands-1), channels[ch].startingPy[i] * shiftFactor * (nrBands-1));
                //System.out.println("shift_end: " + channels[ch].param.dir(i).getPxPy(1)[0] + "/" + channels[ch].param.dir(i).getPxPy(1)[1]);
                //System.out.println();
                //channels[ch].param.dir(i).calcNew(pixelSize);
            }
            */
        }
        startThreads();
    }
    /*
    private void initThreads(int pixelSize) {
        for (int i = 0; i < nrThreads; i++) {
            if (reconThreads[i] != null) reconThreads[i].interrupt();
            reconThreads[i] = new ReconstructionThread(avf);
            if (autostart) {
                reconThreads[i].start();
                Tool.trace("Started recon thread: " + i);
            }
        }

        // precompute filters for all threads
        if (fut != null) fut.interrupt();
        fut = new FilterUpdateThread();
        for (int ch = 0; ch < nrChannels; ch++) {
            channels[ch].param.setPxlSize(pixelSize);
            fut.setupFilters(ch);
        }

        if (prt != null) prt.interrupt();
        prt = new ParameterRefitThread();

        if (autostart) {
            fut.start();
            prt.start();
            Tool.trace("Started parameter fit/update thread: ");
        }
    }
    */
    private void initLatestReconVec() {
        for (int c = 0; c < nrChannels; c++) {
            latestReconVec[c] = avf.createReal2D(width * 2, height * 2);
            latestReconLock[c] = new ReentrantLock();
        }
    }

    // ---- MGMT ----
    /**
     * Query how many images are queued for reconstruction
     */
    public int nrInReconQueue() {
        return imgsToReconstruct.size();
    }

    public int maxInReconQueue() {
        return nrThreads * 16;
    }

    public int nrInWidefieldQueue() {
        return finalWidefield.size();
    }

    public int maxInWidefieldQueue() {
        return nrThreads * 16;
    }

    public int nrInFinalQueue() {
        return finalRecon.size();
    }

    public int maxInFinalQueue() {
        return nrThreads * 16;
    }

    /**
     * Query and reset how many images where missed
     */
    public int nrMissedImages() {
        int ret = missedDueToQueueFull;
        missedDueToQueueFull = 0;
        return ret;
    }

    /**
     * A single reconstruction thread, pulling raw images from the queue and
     * reconstructing them to a final image
     */
    private class ReconstructionThread extends Thread {

        final VectorFactory avf;
        // filter vectors (generated thread-local, so they live on the correct GPU)
        final Vec2d.Cplx[] otfVector;
        final Vec2d.Real dampBorder;
        final Vec2d.Real[] apoVector, wienDenom;

        Vec2d.Cplx[] fullResult;
        Vec2d.Cplx[] widefield;

        Vec2d.Cplx[][][] inFFT;
        Vec2d.Cplx[][][] separate;
        Vec2d.Cplx[][][] shifted;

        int maxRecon = 0;

        final int band2 = nrBands * 2 - 1;

        /**
         * pre-allocate all the vectors
         */
        ReconstructionThread(VectorFactory v) {
            avf = v;
            otfVector = avf.createArrayCplx2D(nrChannels, width, height);
            dampBorder = avf.createReal2D(width, height);
            apoVector = avf.createArrayReal2D(nrChannels, 2 * width, 2 * height);
            wienDenom = avf.createArrayReal2D(nrChannels, 2 * width, 2 * height);

            fullResult = avf.createArrayCplx2D(nrChannels, 2 * width, 2 * height);
            widefield = avf.createArrayCplx2D(nrChannels, width, height);
            inFFT = avf.createArrayCplx2D(
                    nrChannels, nrDirs, nrPhases, width, height);
            separate = avf.createArrayCplx2D(
                    nrChannels, nrDirs, band2, width, height);
            shifted = avf.createArrayCplx2D(
                    nrChannels, nrDirs, band2, 2 * width, 2 * height);
        }

        public void run() {

            // vectors for intermediate results
            Tool.Timer tAll = Tool.getTimer();
            int reconCount = 0;

            // run the reconstruction loop
            while (!stopReconThreads) {

                // retrieve images from queue
                short[][][] imgs = null;
                try {
                    imgs = imgsToReconstruct.poll(1000, TimeUnit.MILLISECONDS);
                    if (imgs == null) continue;
                } catch (InterruptedException e) {
                    Tool.trace("Thread interrupted, frame missed");
                }
                tAll.start();

                // zero the collecting vectors
                for (int c = 0; c < nrChannels; c++) {
                    widefield[c].zero();
                    fullResult[c].zero();
                }

                // generate result vectors cache on the CPU
                Vec2d.Real[] cpuWidefield
                        = Vec.getBasicVectorFactory().createArrayReal2D(
                                nrChannels, width, height);

                Vec2d.Real[] cpuRes
                        = Vec.getBasicVectorFactory().createArrayReal2D(
                                nrChannels, 2 * width, 2 * height);

                // run all input through fft
                for (int c = 0; c < nrChannels; c++) {

                    int count = 0;

                    // REMARK: Change order of these loops to
                    // set if 'angle then phase' or 'phase then angle'
                    // in input. Remember to do this ALSO in the
                    // parameter estimation
                    for (int a = 0; a < nrDirs; a++) {
                        for (int p = 0; p < nrPhases; p++) {

                            short[] inImg = imgs[c][count++];
                            inFFT[c][a][p].setFrom16bitPixels(inImg);

                            // fade borders
                            inFFT[c][a][p].times(dampBorder);
                            // TODO: this would be a place to add the compensation 
                            // add them up to widefield
                            widefield[c].add(inFFT[c][a][p]);
                            inFFT[c][a][p].fft2d(false);
                        }
                    }

                    // copy back wide-field
                    widefield[c].scal(1.f / (nrDirs * nrPhases));
                    cpuWidefield[c].copy(widefield[c]);
                    // registers wide-fild images
                    if (Registration.isWidefield()) {
                        try {
                            Registration reg = Registration.getRegistration(channels[c].label);
                            cpuWidefield[c] = reg.registerWfImage(cpuWidefield[c]);
                        } catch (NoSuchFieldException ex) {
                        }
                    }
                }

                finalWidefield.offer(cpuWidefield);
                

                // loop channel
                for (int channel = 0; channel < nrChannels; channel++) {

                    // loop pattern directions
                    for (int angIdx = 0; angIdx < nrDirs; angIdx++) {
                        final SimParam.Dir par = getChannel(channel).param.dir(angIdx);

                        // ----- Band separation & OTF multiplication -------
                        BandSeparation.separateBands(inFFT[channel][angIdx], separate[channel][angIdx],
                                par.getPhases(), par.nrBand(), par.getModulations());

                        for (int i = 0; i < band2; i++) {
                            separate[channel][angIdx][i].timesConj(otfVector[channel]);
                        }

                        // ------- Shifts to correct position ----------
                        // band 0 is DC, so does not need shifting, only a bigger vector
                        SimUtils.placeFreq(separate[channel][angIdx][0], shifted[channel][angIdx][0]);

                        // higher bands need shifting
                        for (int b = 1; b < par.nrBand(); b++) {
                            int pos = b * 2, neg = (b * 2) - 1;	// pos/neg contr. to band

                            SimUtils.pasteAndFourierShift(
                                    separate[channel][angIdx][pos], shifted[channel][angIdx][pos],
                                    par.px(b), par.py(b), true);
                            SimUtils.pasteAndFourierShift(
                                    separate[channel][angIdx][neg], shifted[channel][angIdx][neg],
                                    -par.px(b), -par.py(b), true);
                        }

                        // sum up to full result 
                        for (int b = 0; b < band2; b++) {
                            fullResult[channel].add(shifted[channel][angIdx][b]);
                        }

                    } // end direction loop

                    // apply wiener filter and APO
                    fullResult[channel].times(wienDenom[channel]);
                    fullResult[channel].times(apoVector[channel]);

                    fullResult[channel].fft2d(true);
                    cpuRes[channel].copy(fullResult[channel]);
                    // registers reconstructed images

                    latestReconLock[channel].lock();
                    try {
                        latestReconVec[channel].copy(cpuRes[channel]);
                    } finally {
                        latestReconLock[channel].unlock();
                    }
                    if (Registration.isRecon()) {
                        try {
                            Registration reg = Registration.getRegistration(channels[channel].label);
                            cpuRes[channel] = reg.registerReconImage(cpuRes[channel]);
                        } catch (NoSuchFieldException ex) {
                            //do nothing
                        }
                    }
                } // end per-channel loop

                finalRecon.offer(cpuRes);

                //finalImages.offer(res);
                tAll.hold();

                // some feedback
                reconCount++;

                if (maxRecon > 0 && reconCount >= maxRecon) {
                    break;
                }

                if (reconCount % 10 == 0) {
                    int rawImgs = nrChannels * nrDirs * nrPhases;
                    Tool.tell(String.format(
                            "reconst:  #%5d %7.2f ms/fr %7.2f ms/raw %7.2f fps(hr) %7.2f fps(raw)",
                            reconCount, tAll.msElapsed() / 10, tAll.msElapsed() / (10 * rawImgs),
                            1000. / (tAll.msElapsed() / 10.),
                            1000. / (tAll.msElapsed() / (10. * rawImgs))));
                    tAll.stop();
                }

            }
        }

    }

    /**
     * Parameter fitter, on CPU
     */
    public class ParameterRefitThread extends Thread {

        Vec2d.Cplx[][] inFFT;
        Vec2d.Cplx[] separate;
        short[][] currentImage;

        Vec2d.Real borderDampen;

        public ParameterRefitThread() {
            final int band2 = nrBands * 2 - 1;
            VectorFactory vf = Vec.getBasicVectorFactory();

            inFFT = vf.createArrayCplx2D(nrDirs, nrPhases, width, height);
            separate = vf.createArrayCplx2D(band2, width, height);

            borderDampen = vf.createReal2D(width, height);
            borderDampen.addConst(1);
            SimUtils.fadeBorderCos(borderDampen, 10);
        }

        @Override
        public void run() {

            while (!isInterrupted()) {
                try {
                    Tool.Tuple<Integer, Tool.Callback<SimParam>> a
                            = doParameterRefit.take();

                    int ch = (a.first != null) ? (a.first) : (-1);
                    if (ch >= 0 && ch < nrChannels) {
                        this.doRefit(ch, a.second);
                    }
                } catch (InterruptedException e) {
                    //Tool.trace("Parameter refit interrupted, why?");
                    interrupt();
                    continue;
                }
            }
        }

        /**
         * run a parameter refit on channel # idx
         */
        public void doRefit(final int chIdx, Tool.Callback<SimParam> caller) {

            SimParam sp = getChannel(chIdx).param.duplicate();
            OtfProvider otfPr = sp.otf();

            // copy over the images // TODO: directly after startup, this
            // throws execption, so this check seems not enough
            if (latestImage == null) {
                Tool.trace("No image available yet!");
                return;
            }
            currentImage = latestImage[chIdx];

            // run input FFT
            // REMARK: Change order of these loops to
            // set if 'angle then phase' or 'phase then angle'
            // in input. Remember to do this ALSO in the
            // reconstruction
            // TODO: read in the order from SimParam
            int count = 0;
            for (int a = 0; a < nrDirs; a++) {
                for (int p = 0; p < nrPhases; p++) {
                    short[] inImg = currentImage[count++];
                    inFFT[a][p].setFrom16bitPixels(inImg);
                    inFFT[a][p].times(borderDampen);
                    inFFT[a][p].fft2d(false);
                }
            }

            // TODO: think about merging this with the estimation in "SimAlgorithm"
            // and only implement this once
            // run the parameter fit for each angle
            for (int angIdx = 0; angIdx < sp.nrDir(); angIdx++) {

                SimParam.Dir par = sp.dir(angIdx);

                BandSeparation.separateBands(inFFT[angIdx], separate,
                        0, par.nrBand(), null);

                final int lb = 1;
                final int hb = (par.nrBand() == 3) ? (3) : (1);
                final int nBand = par.nrBand() - 1;

                if (otfPr == null) {
                    Tool.error("problem: OTF provider is null??", true);
                    return;
                }

                // RUN a coarse, pixel-precision fit fisrt		
                Vec2d.Real otfAtt = Vec2d.createReal(sp);
                otfPr.writeAttenuationVector(otfAtt, .99, 0.15 * otfPr.getCutoff(), 0, 0);

                Vec2d.Cplx c0 = separate[0].duplicate();
                Vec2d.Cplx c1 = separate[hb].duplicate();

                c0.times(otfAtt);
                c1.times(otfAtt);

                // compute correlation: ifft, mult. in spatial, fft back
                Transforms.fft2d(c0, true);
                Transforms.fft2d(c1, true);
                c1.timesConj(c0);
                Transforms.fft2d(c1, false);

                // find the highest peak in corr of band0 to highest band 
                // with min dist 0.5*otfCutoff from origin, store in 'param'
                double fitExclude = .5;
                double minDist = fitExclude * otfPr.getCutoff() / sp.pxlSizeCyclesMicron();
                double[] peak = Correlation.locatePeak(c1, minDist);

                // get the sub-pixel position of the peak
                /*
		double [] peak = 
		    Correlation.fitPeak( separate[0], separate[hb], 0, 1, 
			otfPr, -par.px(nBand), -par.py(nBand), 
			0.05, 2.5, null ); */
                peak
                        = Correlation.fitPeak(separate[0], separate[hb], 0, 1,
                                otfPr, -peak[0], -peak[1],
                                0.05, 2.5, null);

                Cplx.Double p1
                        = Correlation.getPeak(separate[0], separate[lb],
                                0, 1, otfPr, peak[0] / nBand, peak[1] / nBand, 0.05);

                Cplx.Double p2
                        = Correlation.getPeak(separate[0], separate[lb],
                                0, 1, otfPr, peak[0], peak[1], 0.05);

                Tool.trace(
                        String.format("FIT, ch %4d, dir %1d  -->"
                                + " x %7.3f y %7.3f p %7.3f (m %7.3f)",
                                getChannel(chIdx).chNumber,
                                angIdx, peak[0], peak[1], p1.phase(), p1.hypot()));

                par.setPxPy(-peak[0], -peak[1]);
                par.setPhaOff(p1.phase());
                par.setModulation(1, p1.hypot());
                //par.setModulation( 2, p2.hypot() );
            }

            sp.setParamTimestamp();

            caller.callback(sp);
        }

    }

    /**
     * This updates all filters by computing their cached values
     */
    class FilterUpdateThread extends Thread {

        @Override
        public void run() {

            while (!isInterrupted()) {
                try {
                    int ch = doFilterUpdate.take();
                    if (ch >= 0 && ch < nrChannels) {
                        this.setupFilters(ch);
                    }
                } catch (InterruptedException e) {
                    //Tool.trace("Filter updater interrupted, why?");
                    interrupt();
                    continue;
                }
            }
        }

        // Setup filter: Create on CPU, then copy to GPU
        void setupFilters(final int chIdx) {

            Tool.Timer t1 = Tool.getTimer();
            Tool.Timer t2 = Tool.getTimer();

            final PerChannel ch = getChannel(chIdx);

            // create the border dampening vector
            Vec2d.Real dampBorder = Vec2d.createReal(width, height);
            dampBorder.addConst(1.f);
            SimUtils.fadeBorderCos(dampBorder, 10);

            // create the OTF
            Vec2d.Cplx otfV = Vec2d.createCplx(width, height);
            ch.param.otf().setAttenuation(ch.attStr, ch.attFWHM);
            ch.param.otf().switchAttenuation(ch.useAttenuation);
            ch.param.otf().writeOtfWithAttVector(otfV, 0, 0, 0);

            // create the Wiener filter denominator
            WienerFilter wFilter = new WienerFilter(ch.param);
            Vec2d.Real wien = wFilter.getDenominator(ch.wienParam);

            // create the Apotization vector
            Vec2d.Real apo = Vec2d.createReal(2 * width, 2 * height);
            ch.param.otf().writeApoVector(apo, 1, 2);

            t1.stop();
            t2.start();

            // then, copy to every GPU thread   // TODO: concurrency sync here?
            for (ReconstructionThread r : reconThreads) {
                r.dampBorder.copy(dampBorder);
                r.dampBorder.makeCoherent();

                r.otfVector[chIdx].copy(otfV);
                r.wienDenom[chIdx].copy(wien);
                r.apoVector[chIdx].copy(apo);
                r.otfVector[chIdx].makeCoherent();
                r.wienDenom[chIdx].makeCoherent();
                r.apoVector[chIdx].makeCoherent();
            }

            t2.stop();

            if (ch.useAttenuation) {
                Tool.trace(String.format(
                        "Updates filters: Ch %4d, AS %5.3f AW %5.2f W: %5.3f, took",
                        ch.chNumber, ch.attStr, ch.attFWHM, ch.wienParam)
                        + t1 + " " + t2);
            } else {
                Tool.trace(String.format(
                        "Updates filters: Ch %4d, -attenuation off-  W: %5.3f, took",
                        ch.chNumber, ch.wienParam) + t1 + " " + t2);
            }

        }
    }

    public Vec2d.Real getLatestReconVec(int channelId) {
        return latestReconVec[channelId];
    }

    /**
     * To run the ReconstructionThreads through the NVidia profiler
     */
    public static void main(String[] args) throws Exception {

        if (args.length < 3) {
            System.out.println("usage: config.xml nrThread nrImages");
            return;
        }

        String wd = System.getProperty("user.dir") + "/accel/";
        System.load(wd + "libcudaimpl.so");
        VectorFactory avf = AccelVectorFactory.getFactory();
        Conf cfg = Conf.loadFile(args[0]);

        ReconstructionRunner rr = new ReconstructionRunner(
                cfg.r().cd("vigor-settings"), avf, new String[]{"568"}, false);

        // warm-up fft
        avf.createCplx2D(512, 512).fft2d(false);
        avf.createCplx2D(1024, 1024).fft2d(false);

        int nrThreads = Integer.parseInt(args[1]);
        int nrCount = Integer.parseInt(args[2]);

        for (int i = 0; i < nrThreads * nrCount * 4; i++) {
            rr.imgsToReconstruct.offer(new short[1][15][512 * 512]);
        }

        Tool.Timer t1 = Tool.getTimer();

        AccelVectorFactory.startProfiler();

        // start the n threads
        for (int i = 0; i < nrThreads; i++) {
            rr.reconThreads[i].maxRecon = nrCount;
            rr.reconThreads[i].start();

        }

        // join the n threads
        for (int i = 0; i < nrThreads; i++) {
            rr.reconThreads[i].join();
        }

        AccelVectorFactory.stopProfiler();

        t1.stop();

        int nrFrames = nrCount * nrThreads;
        Tool.trace("Timing " + t1 + " for " + nrFrames + ": "
                + String.format(" %7.4f fps ", nrFrames * 1000 / t1.msElapsed()));

        System.exit(0);
    }

}
