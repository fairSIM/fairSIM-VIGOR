/*
 * This file is part of Free Analysis and Interactive Reconstruction
 * for Structured Illumination Microscopy (fairSIM).
 *
 * fairSIM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * fairSIM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with fairSIM.  If not, see <http://www.gnu.org/licenses/>
 */
package org.fairsim.transport;

import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.measure.Calibration;
import ij.plugin.HyperStackConverter;
import ij.process.FloatProcessor;
import ij.process.ShortProcessor;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.livemode.ReconstructionRunner;
import org.fairsim.sim_algorithm.SimParam;
import org.fairsim.utils.Base64;
import org.fairsim.utils.Tool;
import org.fairsim.livemode.LiveControlPanel;
import org.fairsim.sim_algorithm.OtfProvider;

/**
 * Class to handle .livestack and .livesim files
 *
 * @author m.lachetta
 */
public class LiveStack {

    private final Header header;
    private final List<ImageWrapper> imgs;
    
    /**
     * private constructor for ImagePlus instance of ImageJ
     *
     * @param ip imageplus instance from .livestack.tif or .livesim.tif
     * @throws IOException
     */
    private LiveStack(ImagePlus ip) throws IOException {
        String base64Header = ip.getInfoProperty().split("encoded header: ")[1];
        header = Header.decode(base64Header);
        imgs = new LinkedList<>();
        ImageStack stack = ip.getStack();
        for (int i = 1; i < stack.getSize() + 1; i++) {
            String info = stack.getSliceLabel(i);
            String encodedIWHeader = info.split("header: ")[1];
            ByteBuffer bbHeader = ImageWrapper.decodeHeader(encodedIWHeader);
            ImageWrapper iw = new ImageWrapper(bbHeader);
            short[] pixels = (short[]) stack.getPixels(i);
            iw.copy(pixels, iw.width(), iw.height());
            imgs.add(iw);
        }
        sortAndFillupStack();
    }
    
    private LiveStack(Header header, List<ImageWrapper> imgs) {
        this.header = header;
        this.imgs = imgs;
        sortAndFillupStack();
    }
    
    public LiveStack(String file, int channel) throws IOException {
        if (file.endsWith(".livesim")) {
            int nrBands = 2;
            int nrDirs = 3;
            int nrPhases = 3;
            int nrZ = 1;
            int nanometerPerPixel = 78;
            FileInputStream fis = new FileInputStream(file);
            imgs = new LinkedList<>();
            while (fis.available() > 0) {
                ImageWrapper iw = readImageWrapper(fis);
                if (iw.pos1() == channel) imgs.add(iw);
            }
            fis.close();
            if (imgs.isEmpty()) throw new IOException("File is empty");
            int width = imgs.get(0).width();
            int height = imgs.get(0).height();
            List<Integer> c = new ArrayList<>();
            for (ImageWrapper iw : imgs) {
                if (!c.contains(iw.pos1())) {
                    c.add(iw.pos1());
                }
            }
            Header.Channel[] channels = new Header.Channel[1];
            channels[0] = new Header.Channel("Camera", "unknown", 0, channel, null);
            header = new Header("fastSIM", "unknown", "OLYMPUS PlanApo 60x/1.45 Oil TIRF inf/0.17",
                    "fastSIM objective", width, height, nrZ, nrPhases, nrDirs, nrBands, -1, -1,
                    -1, 2, 1, nanometerPerPixel, nanometerPerPixel, nanometerPerPixel, channels);
        } else throw new IOException("unknown file extension, expect .livesim");
        imgs.sort(null);
    }

    /**
     * Constructs a livestack instance from an .livestack, .livesim,
     * livestack.tif or livesim.tif file
     * @param file input file
     * @throws IOException 
     */
    public LiveStack(String file) throws IOException {
        if (file.endsWith(".livestack")) {
            FileInputStream fis = new FileInputStream(file);
            header = Header.read(fis);
            imgs = new LinkedList<>();
            while (fis.available() > 0) {
                imgs.add(readImageWrapper(fis));
            }
            fis.close();
            if (imgs.isEmpty()) throw new IOException("File is empty");
            //ls = new LiveStack(h, i);
        } else if (file.endsWith(".livesim")) {
            int nrBands = 2;
            int nrDirs = 3;
            int nrPhases = 3;
            int nrZ = 1;
            int nanometerPerPixel = 78;
            FileInputStream fis = new FileInputStream(file);
            imgs = new LinkedList<>();
            while (fis.available() > 0) {
                imgs.add(readImageWrapper(fis));
            }
            fis.close();
            if (imgs.isEmpty()) throw new IOException("File is empty");
            int width = imgs.get(0).width();
            int height = imgs.get(0).height();
            List<Integer> c = new ArrayList<>();
            for (ImageWrapper iw : imgs) {
                if (!c.contains(iw.pos1())) {
                    c.add(iw.pos1());
                }
            }
            Header.Channel[] channels = new Header.Channel[c.size()];
            for (int i = 0; i < c.size(); i++) {
                channels[i] = new Header.Channel("Camera", "unknown", 0, c.get(i), null);
            }
            header = new Header("fastSIM", "unknown", "OLYMPUS PlanApo 60x/1.45 Oil TIRF inf/0.17",
                    "fastSIM objective", width, height, nrZ, nrPhases, nrDirs, nrBands, -1, -1,
                    -1, 2, 1, nanometerPerPixel, nanometerPerPixel, nanometerPerPixel, channels);
        } else if (file.endsWith(".livestack.tif") || file.endsWith(".livesim.tif")) {
            ImagePlus ip = new Opener().openImage(file);
            LiveStack temp = new LiveStack(ip);
            header = temp.header;
            imgs = temp.imgs;
            if (imgs.isEmpty()) throw new IOException("File is empty");
        } else {
            throw new IOException("unknown file extension, expect .livestack or .livesim");
        }
        sortAndFillupStack();
    }
    
    /**
     * Reads the header of a .livestack file
     * @param file .livestack file
     * @return header instance of the .livestack file
     * @throws IOException 
     */
    public static Header readHeaderFromFile(String file) throws IOException {
        if (file.endsWith(".livestack")) {
            FileInputStream fis = new FileInputStream(file);
            Header h = Header.read(fis);
            fis.close();
            return h;
        } else {
            throw new IOException("unknown file extension, expect .livestack");
        }
    }
    
    /**
     * splits this livestack into two livestacks in time. This livestacks
     * contains the first half, the returned livestack contains the second half.
     * @return the second half of the splitted livestack
     */
    public LiveStack split() {
        List<ImageWrapper> secondHalf = new LinkedList<>();
        
        int iwPerChannel = sortAndFillupStack();
        int nrCh = header.channels.length;
        if (imgs.size() % nrCh != 0) throw new RuntimeException("Missmatch in iwPerChannel or nrCh " + imgs.size() + " " + nrCh);
        
        for (int c = 0; c < nrCh; c++) {
            List<ImageWrapper> channelList = new ArrayList<>();
            for (int i = 0; i < imgs.size(); i++) { //split into channels
                ImageWrapper iw = imgs.get(i);
                if (iw.pos1() == header.channels[c].exWavelength) channelList.add(iw);
            }
            channelList.sort(null);
            if (iwPerChannel != channelList.size()) throw new RuntimeException("Wrong channelList.size()");
            
            for (int i = 0; i < iwPerChannel / 2 + iwPerChannel % 2; i++) {
                ImageWrapper iw = channelList.get(i + iwPerChannel / 2);
                secondHalf.add(iw);
                imgs.remove(iw);
            }
        }
        
        return new LiveStack(header, secondHalf);
    }

    /**
     * class for the livestack metadata
     */
    public static class Header implements Serializable {

        static final long serialVersionUID = 1;
        static final String ILLUMINATIONTIMEUNIT = "us";
        static final String DELAYTIMEUNIT = "ms";
        static final String SYNCDELAYTIMEUNIT = "us";
        static final String PIXELSIZEUNIT = "nm";
        final String microscope, timestamp, sample, objective;
        final int width, height; // in pixels
        final int zSlices;
        final int nrPhases, nrAngles, nrBands; // sim
        final int illuminationTime; // in µs
        final int delayTime; // in ms
        final int syncDelayTime; // in µs
        final int syncFreq; // ammount of sim sequences between syncs
        final int syncMode;
        final float samplePixelSizeX, samplePixelSizeY, samplePixelSizeZ; // int nm
        final Channel[] channels;

        /**
         *
         * @param microscope
         * @param timestamp
         * @param sample
         * @param objective
         * @param width in pixels
         * @param height in pixels
         * @param zSlices amount of z slices
         * @param nrPhases amount of SIM phases
         * @param nrAngles amount of SIM angles
         * @param nrBands amount of SIM bands
         * @param illuminationTime in µs
         * @param delayTime in ms
         * @param syncDelayTime in µs
         * @param samplePixelSizeX in nm
         * @param syncFreq amount of sim sequences between sync frames
         * @param samplePixelSizeY in nm
         * @param samplePixelSizeZ in nm
         * @param channels channel meta data
         */
        public Header(String microscope, String timestamp, String sample,
                String objective, int width, int height, int zSlices,
                int nrPhases, int nrAngles, int nrBands, int illuminationTime, int delayTime, int syncDelayTime, int syncFreq, int syncMode,
                float samplePixelSizeX, float samplePixelSizeY, float samplePixelSizeZ, Channel[] channels) {

            this.microscope = microscope;
            this.timestamp = timestamp; //UTC timestamp in "yyyyMMdd'T'HHmmss" format
            this.sample = sample;
            this.objective = objective;
            this.width = width;
            this.height = height;
            this.zSlices = zSlices;
            this.nrPhases = nrPhases;
            this.nrAngles = nrAngles;
            this.nrBands = nrBands;
            this.illuminationTime = illuminationTime;
            this.delayTime = delayTime;
            this.syncDelayTime = syncDelayTime;
            this.syncFreq = syncFreq;
            this.syncMode = syncMode;
            this.samplePixelSizeX = samplePixelSizeX;
            this.samplePixelSizeY = samplePixelSizeY;
            this.samplePixelSizeZ = samplePixelSizeZ;
            this.channels = channels;
        }

        /**
         * class for channel metadata
         */
        public static class Channel<T> implements Serializable {

            static final long serialVersionUID = 1;
            final String detector, dye;
            final float illuminationPower; // in mW
            final int exWavelength; // wavelength of excitation in nm
            final T perChannel;

            /**
             *
             * @param detector camera type e.g.
             * @param dye which is in the sample for this channel
             * @param illuminationPower in mW
             * @param exWavelength in nm
             */
            public Channel(String detector, String dye, float illuminationPower, int exWavelength, T perChannel) {

                this.detector = detector;
                this.dye = dye;
                this.illuminationPower = illuminationPower;
                this.exWavelength = exWavelength;
                this.perChannel = perChannel;
            }

        }
        
        /**
        * Converts this header instance into a human readable and
        * a BASE64 encoded part
        * @return String representation of this header instance
        */
        public String getStringRepresentation() {
            String info = "";
            info += "microscope: " + microscope + "\n";
            info += "yyyyMMdd'T'HHmmss timestamp: " + timestamp + "\n";
            info += "sample: " + sample + "\n";
            info += "objective: " + objective + "\n";
            info += "width: " + width + " pixels \n";
            info += "height: " + height + " pixels \n";
            info += "zSlices: " + zSlices + "\n";
            info += "nrPhases: " + nrPhases + "\n";
            info += "nrAngles: " + nrAngles + "\n";
            info += "illuminationTime: " + illuminationTime + " µs \n";
            info += "delayTime: " + delayTime + " ms \n";
            info += "syncDelayTime: " + syncDelayTime + "µs \n";
            info += "syncFreq: " + syncFreq + "\n";
            info += "syncMode: " + syncMode + "\n";
            info += "samplePixelSizeX: " + samplePixelSizeX + " nm \n";
            info += "samplePixelSizeY: " + samplePixelSizeY + " nm \n";
            info += "samplePixelSizeZ: " + samplePixelSizeZ + " nm \n";
            for (int c = 0; c < channels.length; c++) {
                info += "channel " + c + ": detector: " + channels[c].detector + "\n";
                info += "channel " + c + ": dye: " + channels[c].dye + "\n";
                info += "channel " + c + ": illuminationPower: " + channels[c].illuminationPower + " mW \n";
                info += "channel " + c + ": exWavelength: " + channels[c].exWavelength + " nm \n";
            }
            try {
                info += "encoded header: " + encode();
            } catch (IOException ex) {
                throw new RuntimeException("this should never happen " + ex);
            }
            return info;
        }

        /**
         * writes this livestack header into an output stream
         *
         * @param os output stream for writing
         * @throws IOException
         */
        void write(OutputStream os) throws IOException {
            ObjectOutputStream oos = new ObjectOutputStream(os);
            oos.writeObject(this);
            oos.flush();
        }

        /**
         * reads and constructs a header instance from an input stream
         *
         * @param is input stream to be read
         * @return header instance
         * @throws IOException
         */
        private static Header read(InputStream is) throws IOException {
            ObjectInputStream ois = new ObjectInputStream(is);
            try {
                return (Header) ois.readObject();
            } catch (ClassNotFoundException ex) {
                throw new IOException(ex);
            }
        }

        /**
         * encodes this header instance as base64 string
         *
         * @return encoded base64 string
         * @throws IOException
         */
        private String encode() throws IOException {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            write(baos);
            baos.close();
            byte[] bytes = baos.toByteArray();
            return Base64.encode(bytes);
        }

        /**
         * decodes a base64 encoded header instance and constructs it
         *
         * @param encoded base64 encoded header instance
         * @return decoded header instance
         * @throws IOException
         */
        private static Header decode(String encoded) throws IOException {
            byte[] bytes = Base64.decode(encoded);
            ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
            return read(bais);
        }
    }

    /**
     * reads an constructs an ImageWrapper from an input stream
     *
     * @param is inputstream to be read
     * @return the read ImageWrapper instance
     * @throws IOException
     */
    private static ImageWrapper readImageWrapper(InputStream is) throws IOException {
        ByteBuffer bbHeader = ImageWrapper.readImageWrapperHeader(is);
        ImageWrapper iw = new ImageWrapper(bbHeader);
        iw.readData(is);
        iw.parseHeader();
        return iw;
    }

    /**
     * saves this livestack instance as tiff file
     * @param outFile target for saving
     * @param dump if true ImageWrapper of livestack instance gets deleted while saving, for conserving memory
     * @return returns the ImagePlus instance of the tif file
     */
    public ImagePlus saveAsTiff(String outFile, boolean dump) {
        ImagePlus ip = convertToImagePlus(dump);
        FileSaver fs = new FileSaver(ip);
        fs.saveAsTiffStack(outFile);
        return ip;
    }

    /**
     * saves this livestack instance as tiff file
     * @param outFile target for saving
     * @return 
     */
    public ImagePlus saveAsTiff(String outFile) {
        return saveAsTiff(outFile, false);
    }

    /**
     * Gets ImageWrapper into [channel][seqNr/time] order and if necessary fills
     * missing frames with black images with seqNr=Long.MAX_VALUE
     * @return amount of ImageWrapper per channel
     */
    private int sortAndFillupStack() {
        int nrCh = header.channels.length;
        int[] chCounter = new int[nrCh];

        // fill missing frames with blackscreens
        for (ImageWrapper iw : imgs) {
            for (int c = 0; c < nrCh; c++) {
                if (iw.pos1() == header.channels[c].exWavelength) {
                    chCounter[c]++;
                }
            }
        }
        int iwPerChannel = 0;
        for (int c : chCounter) {
            if (c > iwPerChannel) {
                iwPerChannel = c;
            }
        }
        for (int c = 0; c < nrCh; c++) {
            if (chCounter[c] < iwPerChannel) {
                int diff = iwPerChannel - chCounter[c];
                for (int i = 0; i < diff; i++) {
                    ImageWrapper iw = new ImageWrapper(header.width, header.height);
                    iw.copy(new short[header.width * header.height], header.width, header.height);
                    iw.setPos012(iw.pos0(), header.channels[c].exWavelength, iw.pos2());
                    iw.setSeqNr(Long.MAX_VALUE);
                    iw.writeHeader();
                    imgs.add(iw);
                }
            }
        }

        imgs.sort(null);
        return iwPerChannel;
    }
    
    /**
     * converts this livestack instance into an ImagePlus instance including all
     * meta data
     * @param dump if true ImageWrapper of livestack instance gets deleted while saving, for conserving memory
     * @return this livestack instance as ImagePlus instance
     */
    public ImagePlus convertToImagePlus(boolean dump) {
        sortAndFillupStack();
        ImageStack is = new ImageStack(header.width, header.height);
        int nrCh = header.channels.length;
        long firstTime = imgs.get(0).timeCamera();
        long secondTime = imgs.get(1).timeCamera();
        int listSize = imgs.size();
        for (int imgCounter = 0; imgCounter < listSize; imgCounter++) {
            ImageWrapper iw = dump ? imgs.remove(0) : imgs.get(imgCounter);
            for (int c = 0; c < nrCh; c++) {
                if (iw.pos1() == header.channels[c].exWavelength) {
                    ShortProcessor sp = new ShortProcessor(iw.width(), iw.height(), iw.getPixels(), null);
                    String sliceLabel = iw.getHeaderAsString();
                    is.addSlice(sliceLabel, sp);
                }
            }
        }
        //System.out.println(is.getSize());
        ImagePlus ip = new ImagePlus("", is);
        ip = HyperStackConverter.toHyperStack(ip, nrCh, header.zSlices, is.getSize() / nrCh / header.zSlices, "xyztc", "color");
        ip.setProperty("Info", header.getStringRepresentation());
        Calibration calibration = ip.getCalibration();
        calibration.setUnit("µm");
        calibration.pixelWidth = header.samplePixelSizeX / 1000.0;
        calibration.pixelHeight = header.samplePixelSizeY / 1000.0;
        calibration.pixelDepth = header.samplePixelSizeZ / 1000.0;
        calibration.setTimeUnit("ms");
        calibration.frameInterval = (secondTime - firstTime) / 1000.0;
        calibration.fps = 1 / calibration.frameInterval * 1000.0;
        return ip;
    }
    
    /**
     * constructs and initializes a sim reconstruction runner for this
     * livestack instance
     * @return a SimReconstructor for this livestack instance
     */
    SimReconstructor loadSimReconstructor(VectorFactory vf) {
        ReconstructionRunner.PerChannel[] pc = new ReconstructionRunner.PerChannel[header.channels.length];
        for (int i = 0; i < header.channels.length; i++) {      //get reconstructionParameters from LiveReconstruction
            if (header.channels[i].perChannel instanceof ReconstructionRunner.PerChannel) {
                pc[i] = (ReconstructionRunner.PerChannel) header.channels[i].perChannel;
            } else {
                throw new RuntimeException("need instance of ReconstructionRunner.PerChannel");
            }
        }
        return new SimReconstructor(vf, header.width, header.nrPhases, header.nrAngles, header.nrBands, pc);
    }
    
    SimReconstructor loadSimReconstructor(VectorFactory vf, int chIdx, double wienParam, double attStr, double attFWHM, double apoCutOff, boolean useAtt) {
        SimReconstructor sr = loadSimReconstructor(vf);
        sr.setFilterParameters(chIdx, wienParam, attStr, attFWHM, apoCutOff, useAtt);
        return sr;
    }

    /**
     * reconstructs this livestack images via the sim parameters of the
     * livestack header
     * @return the reconstructed stack
     */
    public ReconStack reconstructByHeader(SimReconstructor sr) {
        ImageWrapper[][][] iws = getSimSequences();
        SimReconstructor recRunner = sr;
        return recRunner.reconstruct(iws);
    }
    
    /**
     * reconstructs this livestack images via fiting individual parameters for 
     * each sim sequence
     * @return the reconstructed stack
     */
    public ReconStack reconstructByIndividualFit(SimReconstructor sr) {
        SimReconstructor recRunner = sr;
        ImageWrapper[][][] iws = getSimSequences();
        ReconStack reconStack = null;
        int nrTime = iws.length;
        int nrCh = iws[0].length;
        int nrPa = iws[0][0].length;
        for (int t = 0; t < nrTime; t++) {
            short[][][] pixels = new short[nrCh][nrPa][];
            ImageWrapper[][][] recIw = new ImageWrapper[1][nrCh][nrPa];
            for (int c = 0; c < nrCh; c++) {
                for (int pa = 0; pa < nrPa; pa++) {
                    pixels[c][pa] = iws[t][c][pa].getPixels();
                    recIw[0][c][pa] = iws[t][c][pa];
                }
            }
            for (int c = 0; c < nrCh; c++) {
                recRunner.reFit(pixels, c);
            }
            ReconStack nextReconStack = recRunner.reconstruct(recIw);
            if (t == 0) reconStack = nextReconStack;
            else reconStack.add(nextReconStack);
        }
        return reconStack;
    }
    
    /**
     * reconstructs this livestack images via fiting the parameters for a
     * specified sequence
     * @param time
     * @return the reconstructed stack
     */
    public ReconStack reconstructByFit(SimReconstructor sr, int time) {
        SimReconstructor recRunner = sr;
        ImageWrapper[][][] iws = getSimSequences();
        int nrCh = iws[0].length;
        int nrPa = iws[0][0].length;
        
        short[][][] pixels = new short[nrCh][nrPa][];
        for (int c = 0; c < nrCh; c++) {
            for (int pa = 0; pa < nrPa; pa++) {
                pixels[c][pa] = iws[time][c][pa].getPixels();
            }
        }
        for (int c = 0; c < nrCh; c++) {
            recRunner.reFit(pixels, c);
        }
        
        return recRunner.reconstruct(iws);
    }
    
    /**
     * reconstructs this livestack images via paramets which includes the best
     * modulation depth
     * @return the reconstructed stack
     */
    public ReconStack reconstructByBestFit(SimReconstructor sr) {
        SimReconstructor recRunner = sr;
        ImageWrapper[][][] iws = getSimSequences();
        int nrTime = iws.length;
        int nrCh = iws[0].length;
        int nrPa = iws[0][0].length;        
        double[] maxOfMinModulation = new double[nrCh];
        int[] bestParam = new int[nrCh];
        short[][][] fitPixels = new short[nrCh][nrPa][];
        
        for (int t = 0; t < nrTime; t++) {
            
            double[] minModulation = new double[nrCh];
            for (int c = 0; c < nrCh; c++) {
                for (int pa = 0; pa < nrPa; pa++) {
                    fitPixels[c][pa] = iws[t][c][pa].getPixels();
                }
                minModulation[c] = 2;
                
            }
            
            for (int c = 0; c < nrCh; c++) {
                SimParam sp = recRunner.reFit(fitPixels, c);
                for (int a = 0; a < sp.nrDir(); a++) {
                    SimParam.Dir angle = sp.dir(a);
                    double modulation = angle.getRawModulations()[1];
                    if (modulation < minModulation[c]) {
                        minModulation[c] = modulation;
                    }
                    if (minModulation[c] < 0 || minModulation[c] > 1) throw new RuntimeException("Impossible modulation");
                }
            }
            for (int c = 0; c < nrCh; c++) {
                if (minModulation[c] > maxOfMinModulation[c]) {
                    maxOfMinModulation[c] = minModulation[c];
                    bestParam[c] = t;
                }
            }
        }
        
        for (int c = 0; c < nrCh; c++) {
            for (int pa = 0; pa < nrPa; pa++) {
                    fitPixels[c][pa] = iws[bestParam[c]][c][pa].getPixels();
                }
            recRunner.reFit(fitPixels, c);
        }
        
        return recRunner.reconstruct(iws);
    }
    
    /**
     * class for reconstructed livestacks
     */
    public class ReconStack {
        
        String[][][] iwHeaderStrings; // [t][c][pa]
        float[][][] widefield; //[t][c][xy]
        float[][][] recon; //[t][c][xy]
        
        /**
         * 
         * @param iwHeaderStrings headers of livestack ImageWrappers as Strings
         * [t][c][pa]
         * @param widefield calculated widefield images of the sim sequences [t][c][xy]
         * @param recon reconstructed sim images [t][c][xy]
         */
        private ReconStack(String[][][] iwHeaderStrings, float[][][] widefield, float[][][] recon) {
            if (iwHeaderStrings.length != widefield.length || iwHeaderStrings.length != recon.length)
                throw new RuntimeException("Missmatch in dimensinsion t");
            if (iwHeaderStrings[0].length != widefield[0].length || iwHeaderStrings[0].length != recon[0].length)
                throw new RuntimeException("Missmatch in dimension c");
            if (iwHeaderStrings[0][0].length != header.nrPhases * header.nrAngles)
                throw new RuntimeException("Missmatch in dimension pa");
            if (widefield[0][0].length != header.width * header.height || widefield[0][0].length * 4 != recon[0][0].length)
                throw new RuntimeException("Missmatch in dimension xy");
            
            this.iwHeaderStrings = iwHeaderStrings;
            this.widefield = widefield;
            this.recon = recon;
        }
        
        /**
         * Adds a new ReconStack to this ReconStack
         * @param rs ReconStack which will be add to this
         */
        public void add(ReconStack rs) {
            if (rs.iwHeaderStrings[0].length != iwHeaderStrings[0].length)
                throw new RuntimeException("Missmatch in dimension c");
            if (rs.iwHeaderStrings[0][0].length != iwHeaderStrings[0][0].length)
                throw new RuntimeException("Missmatch in dimension pa");
            if (rs.widefield[0][0].length != widefield[0][0].length)
                throw new RuntimeException("Missmatch in wf dimension xy " + rs.widefield[0][0].length + " " + widefield[0][0].length);
            if (rs.recon[0][0].length != recon[0][0].length)
                throw new RuntimeException("Missmatch in recon dimension xy ");
            
            List<String[][]> newSliceLabel = new ArrayList<>();
            List<float[][]> newWidefields = new ArrayList<>();
            List<float[][]> newRecons = new ArrayList<>();
            
            newSliceLabel.addAll(Arrays.asList(iwHeaderStrings));
            newSliceLabel.addAll(Arrays.asList(rs.iwHeaderStrings));
            iwHeaderStrings = newSliceLabel.toArray(iwHeaderStrings);
            
            newWidefields.addAll(Arrays.asList(widefield));
            newWidefields.addAll(Arrays.asList(rs.widefield));
            widefield = newWidefields.toArray(widefield);
            
            newRecons.addAll(Arrays.asList(recon));
            newRecons.addAll(Arrays.asList(rs.recon));
            recon = newRecons.toArray(recon);
        }
        
        /**
         * Saves the reconstructed images of this reconstack as tif including
         * all meta data
         * @param outFile target for saving
         * @return the saved ImagePlus instance
         */
        public ImagePlus saveReconAsTiff(String outFile) {
            int nrTime = recon.length;
            int nrCh = header.channels.length;
            int nrPa = header.nrPhases * header.nrAngles;
            int nrZ = header.zSlices;
            int width = header.width * 2;
            int height = header.height * 2;
            ImageStack is = new ImageStack(width, height);
            for (int t = 0; t < nrTime; t++) {
                for (int c = 0; c < nrCh; c++) {
                    FloatProcessor sp = new FloatProcessor(width, height, recon[t][c]);
                    String label = iwHeaderStrings[t][c][0];
                    for (int pa = 1; pa < nrPa; pa++) label += "\n" + iwHeaderStrings[t][c][pa];
                    is.addSlice(label, sp);
                }
            }
            ImagePlus ip = new ImagePlus("", is);
            if (nrTime > 1 || nrCh > 1) ip = HyperStackConverter.toHyperStack(ip, nrCh, nrZ, is.getSize() / nrCh / nrZ, "xyzct", "color");
            ip.setProperty("Info", header.getStringRepresentation());
            FileSaver fs = new FileSaver(ip);
            if (nrTime > 1 || nrCh > 1) fs.saveAsTiffStack(outFile);
            else fs.saveAsTiff(outFile);
            return ip;
        }
        
        /**
         * Saves the widefild images of this reconstack as tif including
         * all meta data
         * @param outFile target for saving
         * @return the saved ImagePlus instance
         */
        public ImagePlus saveWfAsTiff(String outFile) {
            int nrTime = widefield.length;
            int nrCh = header.channels.length;
            int nrPa = header.nrPhases * header.nrAngles;
            int nrZ = header.zSlices;
            int width = header.width;
            int height = header.height;
            ImageStack is = new ImageStack(width, height);
            for (int t = 0; t < nrTime; t++) {
                for (int c = 0; c < nrCh; c++) {
                    FloatProcessor sp = new FloatProcessor(width, height, widefield[t][c]);
                    String label = iwHeaderStrings[t][c][0];
                    for (int pa = 1; pa < nrPa; pa++) label += "\n" + iwHeaderStrings[t][c][pa];
                    is.addSlice(label, sp);
                }
            }
            ImagePlus ip = new ImagePlus("", is);
            if (nrTime > 1 || nrCh > 1) ip = HyperStackConverter.toHyperStack(ip, nrCh, nrZ, is.getSize() / nrCh / nrZ, "xyzct", "color");
            ip.setProperty("Info", header.getStringRepresentation());
            FileSaver fs = new FileSaver(ip);
            if (nrTime > 1 || nrCh > 1) fs.saveAsTiffStack(outFile);
            else fs.saveAsTiff(outFile);
            return ip;
        }
    }

    /**
     * ReconstructionRunner for livestack files
     */
    class SimReconstructor extends ReconstructionRunner {

        boolean running = false;
        boolean fitting = false;
        SimParam tempSp; //only need for returning sim parameters

        /**
         * 
         * @param imageSizeInPixels width & height should be equal
         * @param nrPhases amount of phases for sim
         * @param nrDirs amount of angles for sim
         * @param nrBands amount of band for sim
         * @param perChannels array of parameters for sim reconstruction
         */
        private SimReconstructor(VectorFactory vf, int imageSizeInPixels, int nrPhases, int nrDirs, int nrBands, PerChannel[] perChannels) {
            super(vf, 1, imageSizeInPixels, nrPhases, nrDirs, nrBands, perChannels);
        }
        
        private void setFilterParameters(int chIdx, double wienParam, double attStr, double attFWHM, double apoCutOff, boolean useAtt) {
            channels[chIdx].setWienParam(attStr);
            channels[chIdx].setAttStr(attStr);
            channels[chIdx].setAttFWHM(attFWHM);
            channels[chIdx].setApoCutOff(apoCutOff);
            channels[chIdx].setUseAttenuation(useAtt);
        }
        
        /**
         * sets a new 2D-OTF
         * @param chIdx channel index 0, 1, 2
         * @param na NA of the optical system
         * @param lambda emission wavelength in nm
         * @param a curvature factor
         */
        private void setOtfFromEstimate(int chIdx, double na, double lambda, double a) {
            SimParam param = channels[chIdx].getParam();
            OtfProvider otf = OtfProvider.fromEstimate(na, lambda, a);
            param.otf(otf);
        }

        /**
         * Thread.sleep with caught InterruptedException
         * @param time time in ms to sleep
         */
        private void sleeping(long time) {
            try {
                Thread.sleep(time);
            } catch (InterruptedException ex) {
                ex.printStackTrace();
                Tool.error("LiveStack.Reconstructor: interrupted sleeping, why?");
            }
        }

        /**
         * checks the dimension of an image against the dimension of this SimReconstructor
         * @param img [channels][phases & angles][xy]
         */
        private void checkDimension(short[][][] img) {
            if (img[0][0].length != width * height || img[0].length != nrPhases * nrDirs || img.length != nrChannels) {
                Tool.error("LiveStack.Reconstructor: Missmatch in dimensions");
            }
        }
        
        /**
         * reconstructs sim sequences
         * @param raws ImageWrapper for the reconstruction [time][channel][phase & angle]
         * @return the reconstructed stack
         */
        private ReconStack reconstruct(ImageWrapper[][][] raws) {
            while (running || fitting) {
                sleeping(50); //wait for fit to finish
            }
            
            if (nrInReconQueue() > 0 || nrInFinalQueue() > 0) {
                throw new RuntimeException("LiveStack.Reconstructor: Queues should be empty");    //check if queues empty
            }
            running = true;
            int nrSimSeq = raws.length;
            int nrCh = raws[0].length;
            int nrPa = raws[0][0].length;
            int nrPixels = raws[0][0][0].width() * raws[0][0][0].height();
            String[][][] iwHeader = new String[nrSimSeq][nrCh][nrPa];
            float[][][] recon = new float[nrSimSeq][nrCh][nrPixels * 4];
            float[][][] widefield = new float[nrSimSeq][nrCh][nrPixels];
            Thread putThread = new Thread(new Runnable() {          //define new thread that pushes images from list "recons" to reconstruction
                public void run() {
                    for (int t = 0; t < nrSimSeq; t++) {
                        short[][][] raw = new short[nrCh][nrPa][];
                        for (int c = 0; c < nrCh; c++) {
                            for (int pa = 0; pa < nrPa; pa++) {
                                ImageWrapper iw = raws[t][c][pa];
                                raw[c][pa] = iw.getPixels();
                                iwHeader[t][c][pa] = iw.getHeaderAsString();
                            }
                        }
                        checkDimension(raw);
                        try {
                            imgsToReconstruct.put(raw);         //push to reconstruction-queue
                        } catch (InterruptedException ex) {
                            ex.printStackTrace();
                            Tool.error("LiveStack.Reconstructor: interrupted putting img, why?");
                        }
                    }
                }
            });
            Thread takeThread = new Thread(new Runnable() { // define new thread: get reconstructed image from reconstruction
                public void run() {
                    for (int t = 0; t < nrSimSeq; t++) {
                        try {
                            Vec2d.Real[] hr = finalRecon.take();
                            Vec2d.Real[] wf = finalWidefield.take();
                            int len = hr.length;
                            if (len != wf.length) throw new RuntimeException("Error in recon & wf channels");
                            for (int c = 0; c < len; c++) {
                                recon[t][c] = hr[c].vectorData();
                                widefield[t][c] = wf[c].vectorData();
                            }
                        } catch (InterruptedException ex) {
                            ex.printStackTrace();
                            Tool.error("LiveStack.Reconstructor: interrupted taking img, why?");
                        }
                    }
                }
            });
            putThread.start();      //start threads
            takeThread.start();
            try {
                putThread.join();
                takeThread.join();
            } catch (InterruptedException ex) {
                ex.printStackTrace();
                Tool.error("LiveStack.Reconstructor: interrupted joining img, why?");
            }
            running = false;
            
            ReconStack result = new ReconStack(iwHeader, widefield, recon);
            return result;
        }

        /**
         * fits parameters for a sim sequence
         * @param fitImg sim sequence for the fit [channel][phase & angle][xy]
         * @param chIdx channel index which should be fit
         * @return resulting parameters of the fit
         */
        private SimParam reFit(short[][][] fitImg, int chIdx) {
            checkDimension(fitImg);
            while (running) {
                sleeping(50);
            }
            fitting = true;
            latestImage = fitImg;
            Tool.Tuple<Integer, Tool.Callback<SimParam>> fitCommand
                    = new Tool.Tuple<Integer, Tool.Callback<SimParam>>(
                            chIdx, new Tool.Callback<SimParam>() {
                        @Override
                        public void callback(SimParam sp) {
                            tempSp = sp.duplicate();
                            channels[chIdx].setParam(sp);
                            fitting = false;
                        }
                    });
            if (!doParameterRefit.offer(fitCommand)) {
                Tool.error("LiveStack.Reconstructor: Offering parameter fit went wrong");
            }
            while (fitting) {
                sleeping(50);
            }
            return tempSp;
        }
    }
    
    /**
     * converts this livestack instance into sim sequences
     * @param syncMode sync mode 0 or 1
     * @return sim sequences of this livestack instance
     */
    public ImageWrapper[][][] getSimSequences() {
        //get global stack information
        int iwPerChannel = sortAndFillupStack();
        int nrCh = header.channels.length;
        int nrPa = header.nrPhases * header.nrAngles;
        int syncMode = header.syncMode;
        if (syncMode != 0 && syncMode != 1) throw new UnsupportedOperationException("Unsupported syncMode " + syncMode);
        if (imgs.size() % nrCh != 0) throw new RuntimeException("Missmatch in iwPerChannel or nrCh " + imgs.size() + " " + nrCh);
        
        List<List<ImageWrapper>> channelImgs = new ArrayList<>(nrCh);
        for (int c = 0; c < nrCh; c++) {
            List<ImageWrapper> channelList = new ArrayList<>();
            for (int i = 0; i < imgs.size(); i++) { //split into channels
                ImageWrapper iw = imgs.get(i);
                if (iw.pos1() == header.channels[c].exWavelength) channelList.add(iw);
            }
            channelList.sort(null);
            iwPerChannel = channelList.size();
            int simFramesBetweenSync = header.nrPhases * header.nrAngles * header.syncFreq;
            long currentTime = 0;
            long lastTime = 0;
            int lastSync = - simFramesBetweenSync;
            List<ImageWrapper> syncFrames = new ArrayList<>();
            for (int i = 0; i < iwPerChannel; i++) { //find syny frames depending on syncMode
                if (i == 0) {
                    currentTime = 0;
                    lastTime = 0;
                    lastSync = - simFramesBetweenSync;
                    syncFrames.clear();
                }
                ImageWrapper iw = channelList.get(i);
                currentTime = iw.timeCamera();
                
                if (syncMode == 0) {
                    if (currentTime - lastTime > header.syncDelayTime) {
                        int diffSync = i - lastSync;
                        if (diffSync != simFramesBetweenSync) {
                            //System.out.println(i + " " + lastSync + " " + simFramesBetweenSync);
                            for (int k = 0; k < diffSync; k++) { //remove broken sequences
                                int remove = i - k;
                                if (remove >= 0) channelList.remove(remove);
                            }
                            i = 0;
                        }
                        lastSync = i;
                        iwPerChannel = channelList.size();
                    }
                } else throw new UnsupportedOperationException("Only syncMode = 0 supported");
                
//                if ((syncMode == 0 && currentTime - lastTime > header.syncDelayTime)
//                        || (syncMode == 1 && ((Math.abs(currentTime - lastTime - 5000) < 16) || (Math.abs(currentTime - lastTime - 12995) < 16)))) {
//                    int diffSync = i - lastSync;
//                    if (diffSync != simFramesBetweenSync + 2 * syncMode) {
//                        System.out.println(i + " " + lastSync + " " + simFramesBetweenSync);
//                        for (int k = 0; k < diffSync; k++) { //remove broken sequences
//                            int remove = i - k;
//                            if (remove >= 0) channelList.remove(remove);
//                        }
//                        i = 0;
//                    }
//                    lastSync = i;
//                    for(int k = 0; k < 2 * syncMode; k++) syncFrames.add(channelList.get(i - k));
//                    iwPerChannel = channelList.size();
//                }
                lastTime = currentTime;
            }
            channelList.removeAll(syncFrames); //delete syncFrames if syncMode == 1
            iwPerChannel = channelList.size();
            int delete = iwPerChannel % simFramesBetweenSync;
            for (int i = 0; i < delete; i++) channelList.remove(iwPerChannel - 1 - i); //remove last broken sequence
            channelImgs.add(channelList);
        }
        
        iwPerChannel = Integer.MAX_VALUE; //adapt list size of channels
        for (List l : channelImgs) {
            if (l.size() < iwPerChannel) iwPerChannel = l.size();
        }
        for (List l : channelImgs) {
            int diff = l.size() - iwPerChannel;
            if (diff > 0) {
                for(int k = 0; k < diff; k++) l.remove(l.size() - 1);
            }
        }

//        for (List l : channelImgs) {
//            if (l.size() != iwPerChannel) throw new RuntimeException("Missmatch in List cleanedImgs size" + l.size() + " " + iwPerChannel);
//        }
        int nrTime = iwPerChannel / nrPa;
        
        ImageWrapper[][][] simSequences = new ImageWrapper[nrTime][nrCh][nrPa];
        for (int c = 0; c < nrCh; c++) { //put sequences into the correct order for the reconstruction
            for (int i = 0; i < iwPerChannel; i++) {
                int pa = i % nrPa;
                int t = i  / nrPa;
                simSequences[t][c][pa] = channelImgs.get(c).get(i);
            }
        }
        return simSequences;
    }
    
    /**
     * creates livesim-style metadata from a livestack/livesim-TIFF file
     *
     * @param outFile_woExtension target for saving w/o ectension
     * @throws IOException
     */
    public void toMeta(String outFile_woExtension) throws IOException {

        for (int c = 0; c < header.channels.length; c++) {
            int count = 0;
            int lambda = header.channels[c].exWavelength;
            String outFile = outFile_woExtension + lambda + ".meta.txt";
            //System.out.print("\t\twriting to " + outFile + " ... ");
            FileWriter metaFile = new FileWriter(outFile);
            metaFile.write("# idxAl idxCh timeCam timeCapture avr seqNr HeaderBASE64\n");
            for (int allCount = 0; allCount < imgs.size(); allCount++) {
                ImageWrapper iw = imgs.get(allCount);
                if (iw.pos1() == header.channels[c].exWavelength) {

                    //compute average
                    short[] pxls = iw.getPixels();
                    double avr = 0;
                    for (short p : pxls) {
                        avr += p;
                    }
                    avr /= pxls.length;

                    // BASE64 encode the header
                    byte[] header = new byte[128];
                    System.arraycopy(iw.refBuffer(), 0, header, 0, 128);
                    String headerBase64 = Base64.encode(header);

                    String metaLine = String.format(" %8d  %8d  %18d %18d %8.2f %16d %s \n", allCount + 1, count + 1, iw.timeCamera(), iw.timeCapture(), avr, iw.seqNr(), headerBase64);
                    metaFile.write(metaLine);
                    count++;
                }
            }
            metaFile.close();
            //System.out.println(" writing done");
        }
    }

    /**
     * for testing
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        boolean tif = false;
        boolean meta = false;
        boolean rec = false;
        String outFile;

        if (args.length > 1) {
            if (args[0].compareToIgnoreCase("reconstruct") == 0) {
                rec = true;
            } else if (args[0].compareToIgnoreCase("livesim2tif") == 0) {
                tif = true;
            } else if (args[0].compareToIgnoreCase("livesim2meta") == 0) {
                meta = true;
            } else if (args[0].compareToIgnoreCase("livesim2both") == 0) {
                tif = true;
                meta = true;
            }

            if (!(tif || meta || rec)) {
                System.out.println("First parameter should be {reconstruct, livesim2tif, livesim2meta, livesim2both}");
            } else if ((rec) && (args.length != 5)) {
                // more Text
            } else if ((tif || meta) && args.length != 3) {
                System.out.println("# where \"Operation\" is \"livesim2tif\", \"livesim2meta\" or \"livesim2both");
            }
//            System.out.println("# Reconstruction usage: reconstruct Input-file  Output-folder Option1 Option1 Option3");
//            System.out.println("# more text");
//            System.exit(2);
        }

        File file = new File(args[1]);
        File outdir = new File(args[2]);

        if (!file.exists()) {
            System.out.println("File doesn't exist: " + file);
            System.exit(3);
        }
        if (!file.canRead()) {
            System.out.println("Cant' read file: " + file);
            System.exit(4);
        }
        if (!outdir.exists()) {
            System.out.println("Directory doesn't exist: " + outdir);
            System.exit(5);
        }
        if (!outdir.canWrite()) {
            System.out.println("can't write to directory " + outdir);
            System.exit(6);
        }

        System.out.print("\topening " + file.getAbsolutePath() + " ...");
        LiveStack ls = new LiveStack(file.getAbsolutePath());
        System.out.println(" done");
        if (rec) {
            
            String reconFile = outdir.getAbsolutePath() + File.separator + file.getName() + ".rec.tif";
            String wfFile = outdir.getAbsolutePath() + File.separator + file.getName() + ".wf.tif";
            System.out.println("\treconstructing ...");
            
            VectorFactory vf = LiveControlPanel.loadVectorFactory();
            SimReconstructor sr = ls.loadSimReconstructor(vf);
            //sr.setOtfFromEstimate(0, 1.4, 600, 0.3);
            //sr.setFilterParameters(0, 0, 0, 0, 0, true);
            ReconStack reconStack = ls.reconstructByBestFit(sr);
            reconStack.saveReconAsTiff(reconFile);
            reconStack.saveWfAsTiff(wfFile);
            
            System.out.println(" saving done");

        } else {
            System.out.print("Extracting ");
            if (tif) {
                System.out.print("tif");
            }
            if (tif && meta) {
                System.out.print(" and ");
            }
            if (meta) {
                System.out.print("meta");
            }
            System.out.println(" from " + args[1]);
            if (tif) {
                outFile = outdir.getAbsolutePath() + File.separator + file.getName() + ".tif";
                System.out.print("\tsaving tif as: " + outFile + " ...");
                ls.saveAsTiff(outFile, true);
                System.out.println(" saving done");
            }
            if (meta) {
                outFile = outdir.getAbsolutePath() + File.separator + file.getName().replaceAll(".livesim$ || .livestack$", "");
                System.out.println("\tsaving meta-file as: " + outFile + " ...");
                ls.toMeta(outFile);
                System.out.println("\tdone saving meta");
            }
        }
        System.out.println("done\n");
        System.exit(0);
    }
}
