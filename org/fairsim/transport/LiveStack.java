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
import ij.io.Opener;
import ij.plugin.HyperStackConverter;
import ij.process.ShortProcessor;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import static java.lang.Boolean.FALSE;
import static java.lang.Boolean.TRUE;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import loci.common.services.DependencyException;
import loci.common.services.ServiceException;
import org.fairsim.utils.Base64;
import org.fairsim.utils.Tool;
/*
 * import loci.common.services.ServiceFactory;
 * import loci.formats.ImageReader;
 * import loci.formats.meta.IMetadata;
 * import loci.formats.services.OMEXMLService;
 * import loci.formats.out.OMETiffWriter;
 */

import loci.common.services.ServiceFactory;
import loci.formats.out.OMETiffWriter;
import loci.formats.*;
import loci.formats.meta.IMetadata;
import loci.formats.services.OMEXMLService;
import ome.xml.model.enums.*;

import ome.units.UNITS;
import ome.units.quantity.*;
import org.fairsim.accel.AccelVectorFactory;
import org.fairsim.linalg.Vec;
import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.livemode.ReconstructionRunner;
import org.fairsim.sim_algorithm.SimParam;

/**
 * Class for converting .livestack data to other formats like tiff
 *
 * @author m.lachetta
 */
public class LiveStack {

    private final Header header;
    private final List<ImageWrapper> imgs;
    
    private LiveStack(Header header, List<ImageWrapper> imgs) {
        this.header = header;
        this.imgs = imgs;
    }

    /**
     * private constructor for livestack InputStreams use the open method for
     * creating a livestack instance
     *
     * @param is InputStream for the livestack instance
     * @throws IOException
     */
    /*
    private LiveStack(InputStream is) throws IOException {
        header = Header.read(is);
        imgs = new LinkedList<>();
        while (is.available() > 0) {
            imgs.add(readImageWrapper(is));
        }
    }
    */
    /**
     * private constructor for ImagePlus instance of ImageJ
     *
     * @param ip livestack imageplus
     * @throws IOException
     */
    /*
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
    }
    */
    /**
     * creates a livestack instance from a *.livestack or a *.livestack.* file
     *
     * @param file absolute path of the file
     * @return livestack instance of the file
     * @throws IOException
     */
    public static LiveStack open(String file) throws IOException {
        LiveStack ls;
        if (file.endsWith(".livestack")) {
            FileInputStream fis = new FileInputStream(file);
            Header h = Header.read(fis);
            List<ImageWrapper> i = new LinkedList<>();
            while (fis.available() > 0) {
                i.add(readImageWrapper(fis));
            }
            fis.close();
            ls = new LiveStack(h, i);
        } else if (file.endsWith(".livesim")) {
            FileInputStream fis = new FileInputStream(file);
            List<ImageWrapper> iws = new LinkedList<>();
            while (fis.available() > 0) {
                iws.add(readImageWrapper(fis));
            }
            fis.close();
            int width = iws.get(0).width();
            int height = iws.get(0).height();
            List<Integer> c = new ArrayList<>();
            for (ImageWrapper iw : iws) {
                if (!c.contains(iw.pos1())) {
                    c.add(iw.pos1());
                }
            }
            Header.Channel[] channels = new Header.Channel[c.size()];
            for (int i = 0; i < c.size(); i++) {
                channels[i] = new Header.Channel("Camera", "unknown", 0, c.get(i), null);
            }
            Header h = new Header("fastSIM", "unknown", "fastSIM objective",
                    "fastSIM objective", width, height, 1, 3, 3, 2, -1, -1,
                    -1, -1, 79, 79, 79, channels);
            ls = new LiveStack(h, iws);
        } else {
            ImagePlus ip = new Opener().openImage(file);
            String base64Header = ip.getInfoProperty().split("encoded header: ")[1];
            Header h = Header.decode(base64Header);
            List<ImageWrapper> iws = new LinkedList<>();
            ImageStack stack = ip.getStack();
            for (int i = 1; i < stack.getSize() + 1; i++) {
                String info = stack.getSliceLabel(i);
                String encodedIWHeader = info.split("header: ")[1];
                ByteBuffer bbHeader = ImageWrapper.decodeHeader(encodedIWHeader);
                ImageWrapper iw = new ImageWrapper(bbHeader);
                short[] pixels = (short[]) stack.getPixels(i);
                iw.copy(pixels, iw.width(), iw.height());
                iws.add(iw);
            }
            ls = new LiveStack(h, iws);
        }
        return ls;
    }

    /**
     * class for the livestack metadata
     */
    public static class Header implements Serializable {

        static final long serialVersionUID = 1;
        final String microscope, timestamp, sample, objective;
        final int width, height; // in pixels
        final int zSlices;
        final int nrPhases, nrAngles, nrBands; // sim
        final int illuminationTime; // in µs
        final int delayTime; // in ms
        final int syncDelayTime; // in µs
        final int syncFreq; // ammount of sim sequences between syncs
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
         * @param illuminationTime in µs
         * @param delayTime in ms
         * @param samplePixelSizeX in nm
         * @param samplePixelSizeY in nm
         * @param samplePixelSizeZ in nm
         * @param channels channel meta data
         */
        public Header(String microscope, String timestamp, String sample,
                String objective, int width, int height, int zSlices,
                int nrPhases, int nrAngles, int nrBands, int illuminationTime, int delayTime, int syncDelayTime, int syncFreq,
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
            this.samplePixelSizeX = samplePixelSizeX;
            this.samplePixelSizeY = samplePixelSizeY;
            this.samplePixelSizeZ = samplePixelSizeZ;
            this.channels = channels;
        }

        /**
         * class for channel metadata
         */
        public static class Channel implements Serializable {

            static final long serialVersionUID = 1;
            final String detector, dye;
            final float illuminationPower; // in mW
            final int exWavelength; // wavelength of excitation in nm
            final ReconstructionRunner.PerChannel perChannel;

            /**
             *
             * @param detector
             * @param dye
             * @param illuminationPower in mW
             * @param exWavelength in nm
             */
            public Channel(String detector, String dye, float illuminationPower, int exWavelength, ReconstructionRunner.PerChannel perChannel) {

                this.detector = detector;
                this.dye = dye;
                this.illuminationPower = illuminationPower;
                this.exWavelength = exWavelength;
                this.perChannel = perChannel;
            }

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

    public FileSaverThread saveAsOmeTiff(String outFile, boolean dump, int... channels) {
        FileSaverThread fc = new FileSaverThread(outFile, dump, true, false, channels);
        fc.start();
        return fc;
    }

    public FileSaverThread saveAsOmeTiff(String outFile, int... channels) {
        return saveAsOmeTiff(outFile, false, channels);
    }

    /**
     * saves this livestack as tiff file
     *
     * @param outFile absolute file path for saving
     * @param dump dump images for saving heap space
     * @param channels channels which should be saved
     * @return @see FileSaverThread
     */
    public FileSaverThread saveAsTiff(String outFile, boolean dump, int... channels) {
        FileSaverThread fc = new FileSaverThread(outFile, dump, false, true, channels);
        fc.start();
        return fc;
    }

    /**
     * saves this livestack as tiff file
     *
     * @param outFile absolute file path for saving
     * @param channels channels which should be saved
     * @return @see FileSaverThread
     */
    public FileSaverThread saveAsTiff(String outFile, int... channels) {
        return saveAsTiff(outFile, false, channels);
    }

    /**
     * converts a livestack file to a tiff file
     *
     * @param inFile livestack input file
     * @param channels channels which should be converted
     * @throws IOException
     * @throws InterruptedException
     */
    public static void liveStackToTiff(String inFile, int... channels) throws IOException, InterruptedException {
        LiveStack ls = open(inFile);
        String outFile = inFile + ".tif";
        FileSaverThread fst = ls.saveAsTiff(outFile, channels);
        fst.join();
    }

    /**
     * thread for saving files
     */
    public class FileSaverThread extends Thread {

        private String status = "starting";
        private int allCounter, addCounter;
        private final String outFile;
        private final int[] channels;
        private final boolean dump;
        private int nrCh, max;
        private boolean omeTiff, tiff;

        /**
         * private constructor
         *
         * @param outFile absolute file path for saving
         * @param dump dump images for saving heap space
         * @param channels channels which should be saved
         */
        private FileSaverThread(String outFile, boolean dump, boolean omeTiff, boolean tiff, int... channels) {
            this.outFile = outFile;
            this.dump = dump;
            this.omeTiff = omeTiff;
            this.tiff = tiff;

            int nrAllCh = header.channels.length;
            if (channels.length == 0) {
                this.channels = new int[nrAllCh];
                for (int i = 0; i < nrAllCh; i++) {
                    this.channels[i] = header.channels[i].exWavelength;
                }
            } else {
                this.channels = channels;
            }
            for (int channel : this.channels) {
                boolean found = false;
                for (Header.Channel hc : header.channels) {
                    if (channel == hc.exWavelength) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw new RuntimeException("channel " + channel + " not found");
                }
            }

        }

        /**
         *
         * @return status of this thread
         */
        public String getStatus() {
            return status;
        }

        /**
         *
         * @return preparation counter for all images
         */
        public int getAllCounter() {
            return allCounter;
        }

        /**
         *
         * @return preparation counter for selected channel images
         */
        public int getAddCounter() {
            return addCounter;
        }

        private void preparation() {
            status = "preparing";
            nrCh = channels.length;
            int[] chCounter = new int[nrCh];
            for (ImageWrapper iw : imgs) {
                for (int c = 0; c < nrCh; c++) {
                    if (iw.pos1() == header.channels[c].exWavelength) {
                        chCounter[c]++;
                    }
                }
            }
            int maxTemp = 0;
            for (int c : chCounter) {
                if (c > maxTemp) {
                    maxTemp = c;
                }
            }
            max = maxTemp;
            for (int c = 0; c < nrCh; c++) {
                if (chCounter[c] < max) {
                    int diff = max - chCounter[c];
                    for (int i = 0; i < diff; i++) {
                        ImageWrapper iw = new ImageWrapper(header.width, header.height);
                        iw.copy(new short[header.width * header.height], header.width, header.height);
                        iw.setPos012(iw.pos0(), header.channels[c].exWavelength, iw.pos2());
                        iw.writeHeader();
                        imgs.add(iw);
                    }
                }
            }
            imgs.sort(null);
        }

        private void saveTiff() {
            ImageStack is = new ImageStack(header.width, header.height);
            addCounter = 0;
            for (allCounter = 0; allCounter < imgs.size(); allCounter++) {
                ImageWrapper iw = dump ? imgs.remove(0) : imgs.get(allCounter);
                for (int c = 0; c < nrCh; c++) {
                    if (iw.pos1() == channels[c]) {
                        short[] pixels = iw.getPixels();
                        double avr = 0;
                        for (short p : pixels) {
                            avr += p;
                        }
                        avr /= pixels.length;
                        // channel timeCapture avr seqNr HeaderBASE64
                        String sliceLabel = "ch: " + channels[c]
                                + " timestamp: " + iw.timeCamera()
                                + " avr: " + avr
                                + " seqNr " + iw.seqNr()
                                + " header: " + iw.encodeHeader();
                        //System.out.println(ImageWrapper.decodeHeader(iw.encodeHeader()).get(17));
                        ShortProcessor sp = new ShortProcessor(iw.width(), iw.height(), pixels, null);
                        is.addSlice(sliceLabel, sp);
                        addCounter++;
                    }
                }
                if (dump && allCounter % 100 == 0) {
                    System.gc();
                }
            }
            String info = "";
            info += "microscope: " + header.microscope + "\n";
            info += "yyyyMMdd'T'HHmmss timestamp: " + header.timestamp + "\n";
            info += "sample: " + header.sample + "\n";
            info += "objective: " + header.objective + "\n";
            info += "width: " + header.width + " pixels \n";
            info += "height: " + header.height + " pixels \n";
            info += "zSlices: " + header.zSlices + "\n";
            info += "nrPhases: " + header.nrPhases + "\n";
            info += "nrAngles: " + header.nrAngles + "\n";
            info += "illuminationTime: " + header.illuminationTime + " µs \n";
            info += "delayTime: " + header.delayTime + " ms \n";
            info += "samplePixelSizeX: " + header.samplePixelSizeX + " nm \n";
            info += "samplePixelSizeY: " + header.samplePixelSizeY + " nm \n";
            info += "samplePixelSizeZ: " + header.samplePixelSizeZ + " nm \n";
            for (int c = 0; c < nrCh; c++) {
                info += "channel " + c + ": detector: " + header.channels[c].detector + "\n";
                info += "channel " + c + ": dye: " + header.channels[c].dye + "\n";
                info += "channel " + c + ": illuminationPower: " + header.channels[c].illuminationPower + " mW \n";
                info += "channel " + c + ": exWavelength: " + header.channels[c].exWavelength + " nm \n";
            }
            try {
                info += "encoded header: " + header.encode();
            } catch (IOException ex) {
                throw new RuntimeException("this should never happen");
            }
            ImagePlus ip = new ImagePlus("", is);
            ip = HyperStackConverter.toHyperStack(ip, nrCh, header.zSlices, is.getSize() / nrCh / header.zSlices, "xyztc", "color");
            ip.setProperty("Info", info);
            ij.io.FileSaver fs = new ij.io.FileSaver(ip);
            status = "saving";
            fs.saveAsTiffStack(outFile);
            status = "finished";
        }

        private void saveOmeTiff() {
            try {
                OMETiffWriter writer = new OMETiffWriter();
                String id = outFile;
                int dot = id.lastIndexOf(".");
                String outId = (dot >= 0 ? id.substring(0, dot) : id) + ".ome.tif";
                System.out.print("Converting " + id + " to " + outId + " ");
                int pixelType = FormatTools.UINT16;
                ServiceFactory factory = new ServiceFactory();
                OMEXMLService service = factory.getInstance(OMEXMLService.class);
                IMetadata omexmlMeta = service.createOMEXMLMetadata();
                MetadataTools.populateMetadata(omexmlMeta, 0, null, false, "XYZCT",
                        FormatTools.getPixelTypeString(pixelType), header.width, header.height, header.zSlices, channels.length, max, 1);
                omexmlMeta.setDatasetID("date:1", 0);
                omexmlMeta.setDatasetDescription("SIM TIRF:0 b:2 a:3 p:3", 0);
                omexmlMeta.setInstrumentID("fastSIM:1", 0);
                omexmlMeta.setObjectiveID("fastSIM-Olympus:1", 0, 0);
                omexmlMeta.setObjectiveImmersion(Immersion.OIL, 0, 0);
                omexmlMeta.setObjectiveLensNA(1.45, 0, 0);
                omexmlMeta.setObjectiveManufacturer("Olympus", 0, 0);
                omexmlMeta.setObjectiveModel("TIRF", 0, 0);
                omexmlMeta.setObjectiveNominalMagnification(60., 0, 0);
                int planeCount = max * channels.length;
                for (int p = 0; p < planeCount; p++) {
                    omexmlMeta.setPlaneExposureTime(new Time(1234., UNITS.MILLISECOND), 0, p);
                }
                // configure OME-TIFF writer
                writer.setMetadataRetrieve(omexmlMeta);
                writer.setId(outId);
                ImageStack is = new ImageStack(header.width, header.height);
                addCounter = 0;
                for (allCounter = 0; allCounter < imgs.size(); allCounter++) {
                    ImageWrapper iw = dump ? imgs.remove(0) : imgs.get(allCounter);
                    for (int c = 0; c < nrCh; c++) {
                        if (iw.pos1() == channels[c]) {
                            byte[] plane = new byte[2 * header.width * header.height];
                            short[] pixels = iw.getPixels();
                            //                                 System.out.println("pixels.length = "+pixels.length+" firstPixel="+pixels[0]+" lastPixel="+pixels[pixels.length-1]);
                            //                                 System.out.println("plane.length= "+plane.length+"  pixels.length = "+pixels.length);
                            for (int i = 0; i < pixels.length; i++) {
                                plane[2 * i] = (byte) (10 + 10 * (pixels[i] >> 8));
                                plane[2 * i + 1] = (byte) (10 + 10 * (pixels[i] & 0x00FF));
                            }
                            // write plane to output file
                            writer.saveBytes(addCounter, plane);
                            addCounter++;
                            //                                 System.out.println("wrote plane "+p);
                        }
                    }
                }
                writer.close();
                System.out.println(" [done saving as OME-TIFF]");

            } catch (IOException | DependencyException | ServiceException | FormatException ex) {
                Tool.error(ex.toString(), true);
                throw new RuntimeException(ex);
            }
        }

        /**
         * saves this livestack instance as file
         */
        @Override
        public void run() {
            preparation();
            if (omeTiff) {
                saveOmeTiff();
            }
            if (tiff) {
                saveTiff();
            }
        }
    }

    private VectorFactory getVectorFactory() {
        VectorFactory vf = null;
        String hd = System.getProperty("user.home") + "/documents/";
        String library = "libcudaimpl";
        String extension = null;
        Tool.trace("loading " + library + " library from: " + hd);
        String OS = System.getProperty("os.name").toLowerCase();
        if (OS.contains("nix") || OS.contains("nux") || OS.contains("aix")) {
            extension = "so";
        } else if (OS.contains("win")) {
            extension = "dll";
        }
        try {
            if (extension == null) {
                throw new UnsatisfiedLinkError("No unix or windows system found");
            } else {
                System.load(hd + library + "." + extension);
                vf = AccelVectorFactory.getFactory();
            }
        } catch (UnsatisfiedLinkError ex) {
            System.err.println("[fairSIM]: " + ex);
            System.err.println("[fairSIM]: loading not GPU supported version");
            //ex.printStackTrace();
            vf = Vec.getBasicVectorFactory();
        }
        if (vf == null) {
            Tool.error("LiveStack: No VectorFactory loaded", true);
        }
        return vf;
    }

    List<Vec2d.Real[]> reconstruct() {
        ReconstructionRunner.PerChannel[] pc = new ReconstructionRunner.PerChannel[header.channels.length];
        for (int i = 0; i < header.channels.length; i++) {      //get reconstructionParameters from LiveReconstruction
            pc[i] = header.channels[i].perChannel;
        }
        Reconstructor recRunner = new Reconstructor(1, header.width, header.nrPhases, header.nrAngles, header.nrBands, pc);
        return recRunner.reconstruct(imgsToShortList());
    }

    private class Reconstructor extends ReconstructionRunner {

        

        boolean running = false;
        boolean fitting = false;

        Reconstructor(int nrThreads, int imageSizeInPixels, int nrPhases, int nrDirs, int nrBands, PerChannel[] perChannels) {
            super(getVectorFactory(), nrThreads, imageSizeInPixels, nrPhases, nrDirs, nrBands, perChannels);
        }

        void sleeping(long time) {
            try {
                Thread.sleep(time);
            } catch (InterruptedException ex) {
                ex.printStackTrace();
                Tool.error("LiveStack.Reconstructor: interrupted sleeping, why?");
            }
        }

        void checkDimension(short[][][] img) {
            if (img[0][0].length != width * height || img[0].length != nrPhases * nrDirs || img.length != nrChannels) {
                Tool.error("LiveStack.Reconstructor: Missmatch in dimensions");
            }
        }

        List<Vec2d.Real[]> reconstruct(List<ImageWrapper[][]> raws) {
            while (running || fitting) {
                sleeping(50); //wait for fit to finish
            }
            if (nrInReconQueue() > 0 || nrInFinalQueue() > 0) {
                Tool.error("LiveStack.Reconstructor: Queues should be empty");    //check if queues empty
            }
            int nrImgs = raws.size();
            List<Vec2d.Real[]> recons = new ArrayList<>();
            running = true;
            Thread putThread = new Thread(new Runnable() {          //define new thread that pushes images from list "recons" to reconstruction
                public void run() {
                    for (int i = 0; i < nrImgs; i++) {
                        ImageWrapper[][] iwArray = raws.get(i);          //extract next images for reconstruction
                        short[][][] raw = new short[header.channels.length][header.nrPhases * header.nrAngles][];
                        for (int c = 0; c < header.channels.length; c++) {
                            for (int pa = 0; pa < header.nrPhases * header.nrAngles; pa++) {
                                raw[c][pa] = iwArray[c][pa].getPixels();
                            }
                        }
//                        System.out.println(raw==null);
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
                    for (int i = 0; i < nrImgs; i++) {
                        try {
                            recons.add(finalRecon.take());
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
            return recons;
        }

        void reFit(short[][][] fitImg, int chIdx) {
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
                        public void callback(SimParam a) {
                            fitting = false;
                        }
                    });
            if (!doParameterRefit.offer(fitCommand)) {
                Tool.error("LiveStack.Reconstructor: Offering parameter fit went wrong");
            }
            while (fitting) {
                sleeping(50);
            }
        }
    }
    
    //andis preperation code prepares reconstruction [channels][p*a][pixels]
    private List<ImageWrapper[][]> imgsToShortList() {
        List<List<ImageWrapper>> iwListList = new ArrayList<List<ImageWrapper>>();
        int outNr = imgs.size();
        for (int c = 0; c < header.channels.length; c++) {
            List<ImageWrapper> iwList = new ArrayList<>();
            for (ImageWrapper iw : imgs) {
                if (iw.pos1() == header.channels[c].exWavelength) {
                    iwList.add(iw);
                }
            }
            iwList.sort(null);
            iwList = removeSyncs(iwList);
            iwListList.add(iwList);
            outNr = Math.min(outNr, iwList.size());
        }

        List<ImageWrapper[][]> raws = new ArrayList<>();

        for (int out = 0; out < outNr; out++) {
            ImageWrapper[][] raw = new ImageWrapper[header.channels.length][header.nrPhases * header.nrAngles];
            for (int c = 0; c < header.channels.length; c++) {
                for (int i = 0; i < header.nrPhases * header.nrAngles; i++) {
                    raw[c][i] = iwListList.get(c).get(i);
//                  raw[c][i] = blue.remove(0).getPixels();
                }
            }
            raws.add(raw);
        }
        System.out.println("--------------------------- raws.size() = " + raws.size());
        return raws;

    }

    private List<ImageWrapper> removeSyncs(List<ImageWrapper> inList) {
        System.out.println("This shall create a list for each channel, without the syncframes and broken SIM-sequences");

        List<ImageWrapper> outList = new ArrayList<>();
        for (int i = 0; i < inList.size(); i++) {
            outList.add(inList.get(i));
        }

        int nImgs = outList.size();
        System.out.println("    nImgs = " + nImgs);
        int syncFrameDelay = header.syncDelayTime;
        int syncFrameDelayJitter = 14;
        int nrSimFrames = header.nrAngles * header.nrPhases * header.syncFreq;
        int nrSyncFrames = 2;

        //Get timestamps
        long[] timestamps = new long[nImgs];
        for (int i = 0; i < nImgs; i++) {
            timestamps[i] = outList.get(i).timeCamera();
        }

        //find syncframes
        System.out.println("    finding syncframes");
        List<Integer> syncFrameList = findSyncFrames(outList, timestamps, syncFrameDelay, syncFrameDelayJitter);

        //search for sim-sequencs between syncframes, add broken sets to remove-list
        List<Integer> nonSimFrameList = findNonSimFrames(outList, syncFrameList, nrSimFrames, nrSyncFrames);

        //add syncframes to remove-list
        if (nrSyncFrames > 0) {
            for (int i = 0; i < syncFrameList.size(); i++) {
                int s = syncFrameList.get(i);
                for (int j = nrSyncFrames - 1; j >= 0; j--) {
                    if (s - j >= 0) {
                        nonSimFrameList.add(s - j);
                    }
                }
            }
        }

        //remove syncframes and known broken sim-sequences
        System.out.println("    removing syncframes and known broken sim-sequences: ");
        reduce(outList, nonSimFrameList);
        nImgs = outList.size();

        //check sequence numbers
        System.out.println("    checking sequence-numbers of remaining images");
        List<Integer> brokenSeqNrList = checkSeqNr(outList, nrSimFrames);
        System.out.println("    removing newly found broken sim-sequences");

        outList = reduce(outList, brokenSeqNrList);

        System.out.println("prepraring done");
        return outList;
    }

    private List<Integer> findNonSimFrames(List<ImageWrapper> iwList, List<Integer> syncFrameList, int nrSimFrames, int nrSyncFrames) {
        System.out.print("        finding non-SIM-frames... ");
        Collections.sort(syncFrameList);
        int nImgs = iwList.size();
        List<Integer> nonSimFrameList = new ArrayList<>();
        nonSimFrameList.add(0);
        if (((syncFrameList.get(0) + 1 - nrSyncFrames) % nrSimFrames) != 0) {
            nonSimFrameList.set(0, nonSimFrameList.get(0) + 1);
            for (int s = 0; s <= syncFrameList.get(0) - nrSyncFrames; s++) {
                nonSimFrameList.add(s);
                System.out.print(s + ",");
            }
        }
        if (syncFrameList.size() > 1) {
            for (int i = 1; i < syncFrameList.size(); i++) {
                if (((syncFrameList.get(i) - syncFrameList.get(i - 1) - nrSyncFrames) % nrSimFrames) != 0) {
                    nonSimFrameList.set(0, nonSimFrameList.get(0) + 1);
                    for (int s = syncFrameList.get(i - 1) + 1; s <= syncFrameList.get(i) - nrSyncFrames; s++) {
                        nonSimFrameList.add(s);
                        System.out.print(s + ",");
                    }
                }
            }
        }
        int lastSyncFrame = syncFrameList.get(syncFrameList.size() - 1);
        if (((nImgs - 1 - lastSyncFrame) % nrSimFrames) != 0) {
            nonSimFrameList.set(0, nonSimFrameList.get(0) + 1);
            for (int s = lastSyncFrame + 1; s < nImgs; s++) {
                nonSimFrameList.add(s);
                System.out.print(s + ",");
            }
        }
        System.out.println("found " + (nonSimFrameList.size() - 1) + " frames in " + nonSimFrameList.get(0) + " incomplete sequences");
        nonSimFrameList.remove(0);
        return nonSimFrameList;
    }

    private List<Integer> findSyncFrames(List<ImageWrapper> iwList, long[] timestamps, int syncFrameDelay, int syncFrameDelayJitter) {
        int nImgs = iwList.size();
        List<Integer> syncFrameList = new ArrayList<>();
        System.out.print("        found Syncframes: ");
        for (int i = 1; i < nImgs; i++) {
            if (Math.abs(timestamps[i] - timestamps[i - 1] - syncFrameDelay) < syncFrameDelayJitter) {
                syncFrameList.add(i);
                System.out.print(i + ", ");
            }
        }
        System.out.println("done");
        if (syncFrameList.size() == 0) {
            System.err.println("        No Sync-frames found");
            System.exit(1);
        }
        return syncFrameList;
    }

    private List<Integer> checkSeqNr(List<ImageWrapper> iwList, int nrSimFrames) {
        System.out.println("        checking seq Nrs. Mismatches: ");
        int nImgs = iwList.size();
        List<Integer> brokenSeqNrList = new ArrayList<>();
        for (int i = 0; i < nImgs / nrSimFrames; i += nrSimFrames) {
            boolean broken = FALSE;
            for (int j = 0; j < nrSimFrames - 1; j++) {
                if ((iwList.get(i + j).seqNr() - iwList.get(i + j + 1).seqNr()) != -1) {
                    broken = TRUE;
                    System.out.println((i + j) + "=" + iwList.get(i + j).seqNr() + "x" + iwList.get(i + j + 1).seqNr() + "=" + (i + j + 1) + ", ");
                }
            }
            if (broken) {
                for (int j = 0; j < nrSimFrames; j++) {
                    brokenSeqNrList.add(i + j);
                }
            }
        }
        System.out.println("done");
        return brokenSeqNrList;
    }

    private List<ImageWrapper> reduce(List<ImageWrapper> inList, List<Integer> brokenSeqNr) {
        if (brokenSeqNr.size() == 0) {
            System.out.println("        nothing to remove");
            return inList;
        }
        List<ImageWrapper> outList = new ArrayList<>();
        for (int i = 0; i < inList.size(); i++) {
            outList.add(inList.get(i));
        }
        Collections.sort(brokenSeqNr);
        System.out.print("        removing " + brokenSeqNr.size() + " frames from list with length " + outList.size() + ": ");
        for (int i = brokenSeqNr.size() - 1; i >= 0; i--) {
//            System.out.print(red.get(i)+", ");
            outList.remove((int) brokenSeqNr.get(i));
        }
        System.out.println("done, new length: " + outList.size());
        return outList;
    }
    

    /**
     * for testing
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {

        LiveStack ls = open("G:\\vigor-tmp\\U2OS_liveact_MTDR_20170210T102109.livesim");
        System.out.println("opened");
        ls.saveAsTiff("G:\\vigor-tmp\\U2OS_liveact_MTDR_20170210T102109.livesim.tif");
        System.out.println("saved");

        /*
        if (args.length != 24) {
            System.out.println("# Usage:\n\tFolder\n\tOmero-identifier\n\tsimFramesPerSync\n\tsyncFrameInterval\n\tminAvrIntensity\n\tsyncFrameDelay\n\tsyncFrameDelayJitter\n\tnrBands\n\tnrDirs\n\tnrPhases\n\temWavelen\n\totfNA\n\totfCorr\n\tpxSize\n\twienParam\n\tattStrength\n\tattFWHM\n\tbkg(to subtract)\n\tdoAttenuation\n\totfBeforeShift\n\tfindPeak\n\trefinePhase\n\tnrSlices\n\toverwriteFiles");
            return;
        }
        File dir = new File(args[0]);
        File[] foundFiles;
        try {
            foundFiles = dir.listFiles(new FilenameFilter() {
                public boolean accept(File dir, String name) {
                    return name.startsWith(args[1]) && name.endsWith(".livestack");
                }
            });
            System.out.println("found " + foundFiles.length + " files");
            if (foundFiles.length < 1) {
                System.out.println("No files found?");
                return;
            }
            System.out.println(foundFiles[0]);
            System.out.println("opening " + foundFiles[0].getAbsolutePath());
            System.out.println("done");
            LiveStack ls = open(foundFiles[0].getAbsolutePath());
            new ij.io.FileSaver(new ImagePlus("test", new ij.process.FloatProcessor(1024, 1024, ls.reconstruct().get(0)[0].vectorData()))).saveAsTiff(foundFiles[0].getAbsolutePath() + ".tiff");
        } catch (NullPointerException e) {
            System.err.println("File not found_");
            e.printStackTrace();
            System.exit(1);
        }
        System.exit(0);
         */
    }
}
