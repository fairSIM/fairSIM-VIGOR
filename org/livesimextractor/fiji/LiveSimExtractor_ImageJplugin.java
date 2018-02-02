package org.livesimextractor.fiji;

import ij.IJ;
import ij.io.FileSaver;
import ij.ImageStack;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import ij.plugin.PlugIn;
import ij.gui.GenericDialog;
import ij.WindowManager;

import java.io.File;
import java.io.FilenameFilter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;
import javax.swing.JFileChooser;

import org.fairsim.linalg.*;
// import org.fairsim.fiji.ImageVector;
import org.fairsim.fiji.DisplayWrapper;
// import org.fairsim.utils.Tool;
import org.fairsim.utils.Conf;
import org.fairsim.utils.ImageDisplay;
import org.fairsim.sim_algorithm.*;
//import org.fairsim.transport.ImageWrapper;
import org.fairsim.fiji.ImageVector;
import org.fairsim.utils.Base64;
import org.fairsim.utils.Tool;
import org.fairsim.utils.VirtualSubStack;

/**
 * ImageJ plugin to extract SIM sequences from stacks and their meta-data file.
 * Generate the meta-data file (and tiff stack, if required) first with
 * LiveSimConverter.
 */
public class LiveSimExtractor_ImageJplugin implements PlugIn {
    // parameters for sequence extraction

    static int minAvrIntensity;
    static long/*[] */ syncFrameDelay/* = {5000, 12995}*/;
    static int simFrames;

    static int nrBands, nrDirs, nrPhases, nrSlices;
    static double emWavelen, otfNA, otfCorr, pxSize, wienParam, attStrength, attFWHM, bkg;
    static boolean doAttenuation, otfBeforeShift, findPeak, refinePhase, override;

    // List of meta-data
    List<MetaData> allFrameList = new ArrayList<MetaData>();
    List<MetaData> syncFrameList = new ArrayList<MetaData>();
    List<MetaData> cleanSyncFrameList = new ArrayList<MetaData>();

    // Meta data of each frame
    class MetaData {

        int frameNr;
        int sortNr;
        long timeCam;
        long timeCap;
        double average;
        boolean isTimeSyncFrame = false;
        boolean isAvrSyncFrame = false;

        @Override
        public String toString() {
            return String.format(" i: %d tCam: %d tCap %d avr: %f",
                    frameNr, timeCam, timeCap, average);
        }
    }

    class metadataComparator implements Comparator<MetaData> {

        @Override
        public int compare(MetaData md1, MetaData md2) {
            if (md1.timeCam > md2.timeCam) {
                return 1;
            } else if (md1.timeCam < md2.timeCam) {
                return -1;
            }
            return 0;
        }
    }

    // read in the meta-data file
    void readFile(File f) {

        Tool.trace("Reading file: " + f);

        try {
            String line;
            BufferedReader br = new BufferedReader(new FileReader(f));
            int lcount = 0;

            while ((line = br.readLine()) != null) {

                lcount++;

                // skip comments
                if (line.charAt(0) == '#') {
                    continue;
                }

                // parse line
                MetaData md = new MetaData();

                try {
                    Scanner sc = new Scanner(line);
                    sc.useLocale(java.util.Locale.US);
                    sc.nextInt();                   // idxAll
                    md.frameNr = sc.nextInt();  // idxChannel
                    md.timeCam = sc.nextLong();
                    md.timeCap = sc.nextLong();
                    md.average = sc.nextDouble();

                    allFrameList.add(md);
                } catch (java.util.InputMismatchException e) {
                    IJ.log("Input mismatch at line " + lcount);
                    IJ.log("line is: " + line);
                    throw (e);
                }

            }
        } catch (FileNotFoundException e) {
            System.out.println("File not found: " + e);
            e.printStackTrace();
            return;
        } catch (IOException e) {
            System.out.println("IO error: " + e);
            return;
        }

        IJ.log("# read meta-data: " + allFrameList.size());
    }

    // extract the SIM sequence
    void findSyncFrames() {
        Collections.sort(allFrameList, new metadataComparator());
        for (int i = 0; i < allFrameList.size(); i++) {
            allFrameList.get(i).sortNr = i;
        }

        syncFrameList.clear();
        long lastTimeStamp = 0;
        int countTimeFrame = 0;

        // find sync frames
        for (int i = 0; i < allFrameList.size() - simFrames; i++) {
            MetaData md = allFrameList.get(i);
            long curTimeStamp = md.timeCam;

            // version 1 (for camera with precise time-stamp, like PCO)
            if (curTimeStamp - lastTimeStamp - syncFrameDelay > 0) {
                md.isTimeSyncFrame = true;
                syncFrameList.add(md);
                IJ.log("found Syncframe " + i + " " + (curTimeStamp - lastTimeStamp));
                countTimeFrame++;
            }

            lastTimeStamp = curTimeStamp;
        }

        IJ.log("# sync frames (time): " + countTimeFrame);
    }

    // clean up sync frames
    void cleanSyncFrameList() {

        if (syncFrameList.size() < 1) {
            System.err.println("Sync frame list empty");
            System.exit(-1);
        }

        if (syncFrameList.size() == 1) {
            MetaData syncFrame = syncFrameList.get(0);
            cleanSyncFrameList.add(syncFrame);
            for (int i = simFrames; i < allFrameList.size() - simFrames; i += simFrames) {
                MetaData fakeFrame = allFrameList.get(syncFrame.sortNr + i);
                cleanSyncFrameList.add(fakeFrame);
            }
        } else {
            int discardedFrameCount = 0;
            
            MetaData lastEntry = syncFrameList.get(0);
            for (int i = 1; i < syncFrameList.size(); i++) {
                MetaData curEntry = syncFrameList.get(i);
                int distance = curEntry.sortNr - lastEntry.sortNr;

                if (distance % simFrames == 0) {
                    cleanSyncFrameList.add(lastEntry);
                } else {
                    discardedFrameCount++;
                }

                lastEntry = curEntry;
            }
            IJ.log("# Found " + cleanSyncFrameList.size() + " frames, discarded " + discardedFrameCount + " frames");
        }

        if (cleanSyncFrameList.size() < 1) {
            System.err.println("No SIM sequences found");
            System.exit(-1);
        }
    }

    void printReorderStats() {
        int reorderedSeqs = 0;
        MetaData Entry = cleanSyncFrameList.get(0);
        for (int i = 1; i < cleanSyncFrameList.size(); i++) {
            MetaData nextEntry = cleanSyncFrameList.get(i);
            int startFrame = allFrameList.get(Entry.sortNr).frameNr;
            IJ.log("startFrame " + startFrame);
            boolean reordered = false;
            for (int j = 0; j < (allFrameList.get(nextEntry.sortNr).frameNr - allFrameList.get(Entry.sortNr).frameNr); j++) {
                if (allFrameList.get(Entry.sortNr + j).frameNr != startFrame + j) {
                    IJ.log("resorted image " + j + " in sequence from " + allFrameList.get(Entry.sortNr + j).frameNr + " to " + (startFrame + j));
                    reordered = true;
                }
            }
            if (reordered) {
                reorderedSeqs += 1;
            }
            Entry = nextEntry;
        }
        IJ.log("reordered " + reorderedSeqs + "/" + cleanSyncFrameList.size());
    }

    // check the sync frame distance
    void printSyncFrameHistogramm() {

        int[] histogramm = new int[40];
        int lastPos = -1;

        for (MetaData i : syncFrameList) {
            int dist = i.frameNr - lastPos;
            if (lastPos != -1) {
                histogramm[Math.min(dist, histogramm.length - 1)]++;
            }
            lastPos = i.frameNr;
        }

        String log = "";

        for (int i = 0; i < histogramm.length; i++) {
            if (histogramm[i] > 0) {
                log += (String.format(" %2d : ", i));
                for (int j = 0; j < Math.min(histogramm[i], 50); j++) {
                    log += ((j < 48) ? ("*") : ("++"));
                }
                log += "\n";
            }
        }

        IJ.log(log);
    }

    public void gui() {
        GenericDialog gd = new GenericDialog("Syncframe detection");

        String[] illuminationtime = {"1", "2", "10"};
        gd.addRadioButtonGroup("illumination time", illuminationtime, 3, 1, "1");

        gd.showDialog();
        if (gd.wasCanceled()) {
            System.out.println("gd canceled");
            return;
        }

        // ---- get parameters ----
        final String tmp = gd.getNextRadioButton();
        switch (tmp) {
            case "1": {
                syncFrameDelay = 5000;
                simFrames = 18;
                break;
            }
            case "2": {
                syncFrameDelay = 5000;
                simFrames = 18;
                break;
            }
            case "10": {
                syncFrameDelay = 12995;
                simFrames = 9;
                break;
            }
        }
    }

    @Override
    public void run(String arg) {

        if (WindowManager.getCurrentImage() == null) {
            IJ.error("No image selected");
            return;
        }

        ImageStack is = WindowManager.getCurrentImage().getStack();

        JFileChooser metaFs = new JFileChooser();
        int metaFsRet = metaFs.showOpenDialog(null);

        if (metaFsRet != JFileChooser.APPROVE_OPTION) {
            return;
        }

        IJ.log("Opening meta file: " + metaFs.getSelectedFile());
        // parse the meta-data file
        readFile(metaFs.getSelectedFile());

        // gui
        gui();
        // find the sync frames
        findSyncFrames();
        // print the histogram
        //printSyncFrameHistogramm();
        // clean up frame list
        cleanSyncFrameList();
        printReorderStats();

        // convert list
        List<Integer> simPositions = new ArrayList<Integer>();
        for (MetaData md : cleanSyncFrameList) {
            for (int i = 0; i < simFrames; i++) {
                //IJ.log("adding "+md.frameNr+" "+(i+1)+": "+(md.frameNr+i+1));
// 				simPositions.add( md.frameNr + i +1 );
                simPositions.add(allFrameList.get(md.sortNr + i).frameNr);
            }
        }

        // create VirtualSubStack
        ImageStack vss = new VirtualSubStack(is, simPositions);

        // Display stack
        ImagePlus displ = new ImagePlus("SIM substack", vss);
        displ.show();

    }
}
