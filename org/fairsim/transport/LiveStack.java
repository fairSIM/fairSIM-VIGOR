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

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.WindowManager;
import ij.gui.GenericDialog;
import ij.io.Opener;
import ij.plugin.HyperStackConverter;
import ij.plugin.PlugIn;
import ij.process.ShortProcessor;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;
import javax.swing.JFileChooser;
import org.fairsim.accel.AccelVectorFactory;
import org.fairsim.linalg.Vec;
import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.livemode.ReconstructionRunner;
import org.fairsim.sim_algorithm.SimParam;
import org.fairsim.utils.Base64;
import org.fairsim.utils.Tool;
import org.fairsim.utils.VirtualSubStack;
import loci.common.services.DependencyException;
import loci.common.services.ServiceException;
import loci.common.services.ServiceFactory;
import loci.formats.out.OMETiffWriter;
import loci.formats.*;
import loci.formats.meta.IMetadata;
import loci.formats.services.OMEXMLService;
import ome.xml.model.enums.*;
import ome.units.UNITS;
import ome.units.quantity.*;


/**
 * Class for converting .livestack data to other formats like tiff
 *
 * @author m.lachetta
 */
public class LiveStack {

    private final Header header;
    private final List<ImageWrapper> imgs;

    /*
    private LiveStack(Header header, List<ImageWrapper> imgs) {
        this.header = header;
        this.imgs = imgs;
    }
     */
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
    public LiveStack(ImagePlus ip) throws IOException {
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

    public LiveStack(String file) throws IOException {
        if (file.endsWith(".livestack")) {
            FileInputStream fis = new FileInputStream(file);
            header = Header.read(fis);
            imgs = new LinkedList<>();
            while (fis.available() > 0) {
                imgs.add(readImageWrapper(fis));
            }
            fis.close();
            //ls = new LiveStack(h, i);
        } else if (file.endsWith(".livesim")) {
            FileInputStream fis = new FileInputStream(file);
            imgs = new LinkedList<>();
            while (fis.available() > 0) {
                imgs.add(readImageWrapper(fis));
            }
            fis.close();
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
            header = new Header("fastSIM", "unknown", "fastSIM objective",
                    "fastSIM objective", width, height, 1, 3, 3, 2, -1, -1,
                    -1, -1, 79, 79, 79, channels);
            //ls = new LiveStack(h, iws);
        } else if (file.endsWith(".livestack.tif")) {
            ImagePlus ip = new Opener().openImage(file);
            LiveStack temp = new LiveStack(ip);
            header = temp.header;
            imgs = temp.imgs;
        } else {
            throw new IOException("unknown file extension, expect .livestack or .livesim");
        }
    }

    /**
     * creates a livestack instance from a *.livestack or a *.livestack.* file
     *
     * @param file absolute path of the file
     * @return livestack instance of the file
     * @throws IOException
     */
    /*
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
     */
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
        public static class Channel<T> implements Serializable {

            static final long serialVersionUID = 1;
            final String detector, dye;
            final float illuminationPower; // in mW
            final int exWavelength; // wavelength of excitation in nm
            final T perChannel;

            /**
             *
             * @param detector
             * @param dye
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

//    public FileSaverThread saveAsOmeTiff(String outFile, boolean dump, int... channels) {
//        FileSaverThread fc = new FileSaverThread(outFile, dump, true, false, channels);
//        fc.start();
//        return fc;
//    }
//
//    public FileSaverThread saveAsOmeTiff(String outFile, int... channels) {
//        return saveAsOmeTiff(outFile, false, channels);
//    }
//
//    /**
//     * saves this livestack as tiff file
//     *
//     * @param outFile absolute file path for saving
//     * @param dump dump images for saving heap space
//     * @param channels channels which should be saved
//     * @return @see FileSaverThread
//     */
//    public FileSaverThread saveAsTiff(String outFile, boolean dump, int... channels) {
//        FileSaverThread fc = new FileSaverThread(outFile, dump, false, true, channels);
//        fc.start();
//        return fc;
//    }
//
//    /**
//     * saves this livestack as tiff file
//     *
//     * @param outFile absolute file path for saving
//     * @param channels channels which should be saved
//     * @return @see FileSaverThread
//     */
//    public FileSaverThread saveAsTiff(String outFile, int... channels) {
//        return saveAsTiff(outFile, false, channels);
//    }
    public ImagePlus saveAsTiff(String outFile, boolean dump) {
        ImagePlus ip = convertToImagePlus(dump);
        ij.io.FileSaver fs = new ij.io.FileSaver(ip);
        fs.saveAsTiffStack(outFile);
        return ip;
    }

    public ImagePlus saveAsTiff(String outFile) {
        return saveAsTiff(outFile, false);
    }

//    /**
//     * converts a livestack file to a tiff file
//     *
//     * @param inFile livestack input file
//     * @param channels channels which should be converted
//     * @throws IOException
//     * @throws InterruptedException
//     */
//    public static void liveStackToTiff(String inFile, int... channels) throws IOException, InterruptedException {
//        LiveStack ls = open(inFile);
//        String outFile = inFile + ".tif";
//        FileSaverThread fst = ls.saveAsTiff(outFile, channels);
//        fst.join();
//    }
    private int orderImagesAsStack() {
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
        int imgsPerChannel = 0;
        for (int c : chCounter) {
            if (c > imgsPerChannel) {
                imgsPerChannel = c;
            }
        }
        for (int c = 0; c < nrCh; c++) {
            if (chCounter[c] < imgsPerChannel) {
                int diff = imgsPerChannel - chCounter[c];
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
        return imgsPerChannel;
    }

    public ImagePlus convertToImagePlus(boolean dump) {
        orderImagesAsStack();
        ImageStack is = new ImageStack(header.width, header.height);
        int nrCh = header.channels.length;
        for (int imgCounter = 0; imgCounter < imgs.size(); imgCounter++) {
            ImageWrapper iw = dump ? imgs.remove(0) : imgs.get(imgCounter);
            for (int c = 0; c < nrCh; c++) {
                if (iw.pos1() == header.channels[c].exWavelength) {
                    short[] pixels = iw.getPixels();
                    double avr = 0;
                    for (short p : pixels) {
                        avr += p;
                    }
                    avr /= pixels.length;
                    // channel timeCapture avr seqNr HeaderBASE64
                    String sliceLabel = "ch: " + header.channels[c].exWavelength
                            + " timestamp: " + iw.timeCamera()
                            + " avr: " + avr
                            + " seqNr " + iw.seqNr()
                            + " header: " + iw.encodeHeader();
                    ShortProcessor sp = new ShortProcessor(iw.width(), iw.height(), pixels, null);
                    is.addSlice(sliceLabel, sp);
                }
            }
        }

        ImagePlus ip = new ImagePlus("", is);
        ip = HyperStackConverter.toHyperStack(ip, nrCh, header.zSlices, is.getSize() / nrCh / header.zSlices, "xyztc", "color");
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
        ip.setProperty("Info", info);
        return ip;
    }

    @Deprecated
    public int saveOmeTiff(String outFile, boolean dump) throws DependencyException, ServiceException, FormatException, IOException {
        int imgsPerChannel = orderImagesAsStack();
        int nrCh = header.channels.length;
        OMETiffWriter writer = new OMETiffWriter();
        String id = outFile;
        int dot = id.lastIndexOf(".");
        String outId = (dot >= 0 ? id.substring(0, dot) : id) + ".ome.tif";
        int pixelType = FormatTools.UINT16;
        ServiceFactory factory = new ServiceFactory();
        OMEXMLService service = factory.getInstance(OMEXMLService.class);
        IMetadata omexmlMeta = service.createOMEXMLMetadata();
        MetadataTools.populateMetadata(omexmlMeta, 0, null, false, "XYZCT",
                FormatTools.getPixelTypeString(pixelType), header.width, header.height, header.zSlices, nrCh, imgsPerChannel, 1);
        omexmlMeta.setDatasetID("date:1", 0);
        omexmlMeta.setDatasetDescription("SIM TIRF:0 b:2 a:3 p:3", 0);
        omexmlMeta.setInstrumentID("fastSIM:1", 0);
        omexmlMeta.setObjectiveID("fastSIM-Olympus:1", 0, 0);
        omexmlMeta.setObjectiveImmersion(Immersion.OIL, 0, 0);
        omexmlMeta.setObjectiveLensNA(1.45, 0, 0);
        omexmlMeta.setObjectiveManufacturer("Olympus", 0, 0);
        omexmlMeta.setObjectiveModel("TIRF", 0, 0);
        omexmlMeta.setObjectiveNominalMagnification(60., 0, 0);
        int planeCount = imgsPerChannel * nrCh;
        for (int p = 0; p < planeCount; p++) {
            omexmlMeta.setPlaneExposureTime(new Time(1234., UNITS.MILLISECOND), 0, p);
        }
        // configure OME-TIFF writer
        writer.setMetadataRetrieve(omexmlMeta);
        writer.setId(outId);
        ImageStack is = new ImageStack(header.width, header.height);
        int imgCounter;
        for (imgCounter = 0; imgCounter < imgs.size(); imgCounter++) {
            ImageWrapper iw = dump ? imgs.remove(0) : imgs.get(imgCounter);
            for (int c = 0; c < nrCh; c++) {
                if (iw.pos1() == header.channels[c].exWavelength) {
                    byte[] plane = new byte[2 * header.width * header.height];
                    short[] pixels = iw.getPixels();
                    for (int i = 0; i < pixels.length; i++) {
                        plane[2 * i] = (byte) (10 + 10 * (pixels[i] >> 8));
                        plane[2 * i + 1] = (byte) (10 + 10 * (pixels[i] & 0x00FF));
                    }
                    // write plane to output file
                    writer.saveBytes(imgCounter, plane);
                    imgCounter++;
                }
            }
        }
        writer.close();
        return imgCounter;
    }

//    /**
//     * thread for saving files
//     */
//    /
//    public class FileSaverThread extends Thread {
//
//        private String status = "starting";
//        private int allCounter, addCounter;
//        private final String outFile;
//        private final int[] channels;
//        private final boolean dump;
//        private int nrCh, max;
//        private boolean omeTiff, tiff;
//
//        /**
//         * private constructor
//         *
//         * @param outFile absolute file path for saving
//         * @param dump dump images for saving heap space
//         * @param channels channels which should be saved
//         */
//        private FileSaverThread(String outFile, boolean dump, boolean omeTiff, boolean tiff, int... channels) {
//            this.outFile = outFile;
//            this.dump = dump;
//            this.omeTiff = omeTiff;
//            this.tiff = tiff;
//
//            int nrAllCh = header.channels.length;
//            if (channels.length == 0) {
//                this.channels = new int[nrAllCh];
//                for (int i = 0; i < nrAllCh; i++) {
//                    this.channels[i] = header.channels[i].exWavelength;
//                }
//            } else {
//                this.channels = channels;
//            }
//            for (int channel : this.channels) {
//                boolean found = false;
//                for (Header.Channel hc : header.channels) {
//                    if (channel == hc.exWavelength) {
//                        found = true;
//                        break;
//                    }
//                }
//                if (!found) {
//                    throw new RuntimeException("channel " + channel + " not found");
//                }
//            }
//
//        }
//
//        /**
//         *
//         * @return status of this thread
//         */
//        public String getStatus() {
//            return status;
//        }
//
//        /**
//         *
//         * @return preparation counter for all images
//         */
//        public int getAllCounter() {
//            return allCounter;
//        }
//
//        /**
//         *
//         * @return preparation counter for selected channel images
//         */
//        public int getAddCounter() {
//            return addCounter;
//        }
//
//        private void preparation() {
//            status = "preparing";
//            nrCh = channels.length;
//            int[] chCounter = new int[nrCh];
//            for (ImageWrapper iw : imgs) {
//                for (int c = 0; c < nrCh; c++) {
//                    if (iw.pos1() == header.channels[c].exWavelength) {
//                        chCounter[c]++;
//                    }
//                }
//            }
//            int maxTemp = 0;
//            for (int c : chCounter) {
//                if (c > maxTemp) {
//                    maxTemp = c;
//                }
//            }
//            max = maxTemp;
//            for (int c = 0; c < nrCh; c++) {
//                if (chCounter[c] < max) {
//                    int diff = max - chCounter[c];
//                    for (int i = 0; i < diff; i++) {
//                        ImageWrapper iw = new ImageWrapper(header.width, header.height);
//                        iw.copy(new short[header.width * header.height], header.width, header.height);
//                        iw.setPos012(iw.pos0(), header.channels[c].exWavelength, iw.pos2());
//                        iw.writeHeader();
//                        imgs.add(iw);
//                    }
//                }
//            }
//            imgs.sort(null);
//        }
//
//        private void saveTiff() {
//            ImageStack is = new ImageStack(header.width, header.height);
//            addCounter = 0;
//            for (allCounter = 0; allCounter < imgs.size(); allCounter++) {
//                ImageWrapper iw = dump ? imgs.remove(0) : imgs.get(allCounter);
//                for (int c = 0; c < nrCh; c++) {
//                    if (iw.pos1() == channels[c]) {
//                        short[] pixels = iw.getPixels();
//                        double avr = 0;
//                        for (short p : pixels) {
//                            avr += p;
//                        }
//                        avr /= pixels.length;
//                        // channel timeCapture avr seqNr HeaderBASE64
//                        String sliceLabel = "ch: " + channels[c]
//                                + " timestamp: " + iw.timeCamera()
//                                + " avr: " + avr
//                                + " seqNr " + iw.seqNr()
//                                + " header: " + iw.encodeHeader();
//                        //System.out.println(ImageWrapper.decodeHeader(iw.encodeHeader()).get(17));
//                        ShortProcessor sp = new ShortProcessor(iw.width(), iw.height(), pixels, null);
//                        is.addSlice(sliceLabel, sp);
//                        addCounter++;
//                    }
//                }
//                if (dump && allCounter % 100 == 0) {
//                    System.gc();
//                }
//            }
//            String info = "";
//            info += "microscope: " + header.microscope + "\n";
//            info += "yyyyMMdd'T'HHmmss timestamp: " + header.timestamp + "\n";
//            info += "sample: " + header.sample + "\n";
//            info += "objective: " + header.objective + "\n";
//            info += "width: " + header.width + " pixels \n";
//            info += "height: " + header.height + " pixels \n";
//            info += "zSlices: " + header.zSlices + "\n";
//            info += "nrPhases: " + header.nrPhases + "\n";
//            info += "nrAngles: " + header.nrAngles + "\n";
//            info += "illuminationTime: " + header.illuminationTime + " µs \n";
//            info += "delayTime: " + header.delayTime + " ms \n";
//            info += "samplePixelSizeX: " + header.samplePixelSizeX + " nm \n";
//            info += "samplePixelSizeY: " + header.samplePixelSizeY + " nm \n";
//            info += "samplePixelSizeZ: " + header.samplePixelSizeZ + " nm \n";
//            for (int c = 0; c < nrCh; c++) {
//                info += "channel " + c + ": detector: " + header.channels[c].detector + "\n";
//                info += "channel " + c + ": dye: " + header.channels[c].dye + "\n";
//                info += "channel " + c + ": illuminationPower: " + header.channels[c].illuminationPower + " mW \n";
//                info += "channel " + c + ": exWavelength: " + header.channels[c].exWavelength + " nm \n";
//            }
//            try {
//                info += "encoded header: " + header.encode();
//            } catch (IOException ex) {
//                throw new RuntimeException("this should never happen");
//            }
//            ImagePlus ip = new ImagePlus("", is);
//            ip = HyperStackConverter.toHyperStack(ip, nrCh, header.zSlices, is.getSize() / nrCh / header.zSlices, "xyztc", "color");
//            ip.setProperty("Info", info);
//            ij.io.FileSaver fs = new ij.io.FileSaver(ip);
//            status = "saving";
//            fs.saveAsTiffStack(outFile);
//            status = "finished";
//        }
//
//        private void saveOmeTiff() {
//            try {
//                OMETiffWriter writer = new OMETiffWriter();
//                String id = outFile;
//                int dot = id.lastIndexOf(".");
//                String outId = (dot >= 0 ? id.substring(0, dot) : id) + ".ome.tif";
//                System.out.print("Converting " + id + " to " + outId + " ");
//                int pixelType = FormatTools.UINT16;
//                ServiceFactory factory = new ServiceFactory();
//                OMEXMLService service = factory.getInstance(OMEXMLService.class);
//                IMetadata omexmlMeta = service.createOMEXMLMetadata();
//                MetadataTools.populateMetadata(omexmlMeta, 0, null, false, "XYZCT",
//                        FormatTools.getPixelTypeString(pixelType), header.width, header.height, header.zSlices, channels.length, max, 1);
//                omexmlMeta.setDatasetID("date:1", 0);
//                omexmlMeta.setDatasetDescription("SIM TIRF:0 b:2 a:3 p:3", 0);
//                omexmlMeta.setInstrumentID("fastSIM:1", 0);
//                omexmlMeta.setObjectiveID("fastSIM-Olympus:1", 0, 0);
//                omexmlMeta.setObjectiveImmersion(Immersion.OIL, 0, 0);
//                omexmlMeta.setObjectiveLensNA(1.45, 0, 0);
//                omexmlMeta.setObjectiveManufacturer("Olympus", 0, 0);
//                omexmlMeta.setObjectiveModel("TIRF", 0, 0);
//                omexmlMeta.setObjectiveNominalMagnification(60., 0, 0);
//                int planeCount = max * channels.length;
//                for (int p = 0; p < planeCount; p++) {
//                    omexmlMeta.setPlaneExposureTime(new Time(1234., UNITS.MILLISECOND), 0, p);
//                }
//                // configure OME-TIFF writer
//                writer.setMetadataRetrieve(omexmlMeta);
//                writer.setId(outId);
//                ImageStack is = new ImageStack(header.width, header.height);
//                addCounter = 0;
//                for (allCounter = 0; allCounter < imgs.size(); allCounter++) {
//                    ImageWrapper iw = dump ? imgs.remove(0) : imgs.get(allCounter);
//                    for (int c = 0; c < nrCh; c++) {
//                        if (iw.pos1() == channels[c]) {
//                            byte[] plane = new byte[2 * header.width * header.height];
//                            short[] pixels = iw.getPixels();
//                            //                                 System.out.println("pixels.length = "+pixels.length+" firstPixel="+pixels[0]+" lastPixel="+pixels[pixels.length-1]);
//                            //                                 System.out.println("plane.length= "+plane.length+"  pixels.length = "+pixels.length);
//                            for (int i = 0; i < pixels.length; i++) {
//                                plane[2 * i] = (byte) (10 + 10 * (pixels[i] >> 8));
//                                plane[2 * i + 1] = (byte) (10 + 10 * (pixels[i] & 0x00FF));
//                            }
//                            // write plane to output file
//                            writer.saveBytes(addCounter, plane);
//                            addCounter++;
//                            //                                 System.out.println("wrote plane "+p);
//                        }
//                    }
//                }
//                writer.close();
//                System.out.println(" [done saving as OME-TIFF]");
//
//            } catch (IOException | DependencyException | ServiceException | FormatException ex) {
//                Tool.error(ex.toString(), true);
//                throw new RuntimeException(ex);
//            }
//        }
//
//        /**
//         * saves this livestack instance as file
//         */
//        @Override
//        public void run() {
//            preparation();
//            if (omeTiff) {
//                saveOmeTiff();
//            }
//            if (tiff) {
//                saveTiff();
//            }
//        }
//    }
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

    @Deprecated
    public List<float[][]> reconstructSim() {
        ReconstructionRunner.PerChannel[] pc = new ReconstructionRunner.PerChannel[header.channels.length];
        for (int i = 0; i < header.channels.length; i++) {      //get reconstructionParameters from LiveReconstruction
            if (header.channels[i].perChannel instanceof ReconstructionRunner.PerChannel) {
                pc[i] = (ReconstructionRunner.PerChannel) header.channels[i].perChannel;
            } else {
                throw new RuntimeException("need instance of ReconstructionRunner.PerChannel");
            }
        }
        SimReconstructor recRunner = new SimReconstructor(1, header.width, header.nrPhases, header.nrAngles, header.nrBands, pc);
        List<ImageWrapper[][]> shorts = extractSequences();
        return recRunner.reconstruct(shorts);
    }

    private class SimReconstructor extends ReconstructionRunner {

        boolean running = false;
        boolean fitting = false;

        SimReconstructor(int nrThreads, int imageSizeInPixels, int nrPhases, int nrDirs, int nrBands, PerChannel[] perChannels) {
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

        List<float[][]> reconstruct(List<ImageWrapper[][]> raws) {
            while (running || fitting) {
                sleeping(50); //wait for fit to finish
            }
            if (nrInReconQueue() > 0 || nrInFinalQueue() > 0) {
                Tool.error("LiveStack.Reconstructor: Queues should be empty");    //check if queues empty
            }
            int nrImgs = raws.size();
            List<float[][]> recons = new ArrayList<>();
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
                            Vec2d.Real[] reals = finalRecon.take();
                            int len = reals.length;
                            float[][] floats = new float[len][];
                            for (int c = 0; c < len; c++) {
                                floats[c] = reals[c].vectorData();
                            }
                            finalWidefield.take();
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
    private List<ImageWrapper[][]> extractSequences() {
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
            System.out.println("outNr = " + outNr);
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
        System.out.println("    outList.size = " + outList.size());
        nImgs = outList.size();

        //check sequence numbers
        System.out.println("    checking sequence-numbers of remaining images");
        List<Integer> brokenSeqNrList = checkSeqNr(outList, nrSimFrames);
        System.out.println("    removing newly found broken sim-sequences");
        reduce(outList, brokenSeqNrList);
        System.out.println("    outList.size = " + outList.size());

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
        System.out.print("        checking seq Nrs. Mismatches: ");
        int nImgs = iwList.size();
        List<Integer> brokenSeqNrList = new ArrayList<>();
        for (int i = 0; i < nImgs / nrSimFrames; i += nrSimFrames) {
            boolean broken = false;
            for (int j = 0; j < nrSimFrames - 1; j++) {
                if ((iwList.get(i + j).seqNr() - iwList.get(i + j + 1).seqNr()) != -1) {
                    broken = true;
                    System.out.println((i + j) + "=" + iwList.get(i + j).seqNr() + "x" + iwList.get(i + j + 1).seqNr() + "=" + (i + j + 1) + ", ");
                }
            }
            if (broken) {
                for (int j = 0; j < nrSimFrames; j++) {
                    brokenSeqNrList.add(i + j);
                }
            }
        }
//        System.out.println("done");
        return brokenSeqNrList;
    }

    private /*List<ImageWrapper>*/ void reduce(List<ImageWrapper> inList, List<Integer> brokenSeqNr) {
        if (brokenSeqNr.size() == 0) {
            System.out.println("        nothing to remove");
//            return inList;
        }
//        List<ImageWrapper> outList = new ArrayList<>();
//        for (int i = 0; i < inList.size(); i++) {
//            outList.add(inList.get(i));
//        }
        Collections.sort(brokenSeqNr);
        System.out.print("        removing " + brokenSeqNr.size() + " frames from list with length " + inList.size() + ": ");
        for (int i = brokenSeqNr.size() - 1; i >= 0; i--) {
//            System.out.print(red.get(i)+", ");
            inList.remove((int) brokenSeqNr.get(i));
        }
        System.out.println("done, new length: " + inList.size());
//        return outList;
    }

    /**
     * creates livesim-style metadata from a livestack/livesim-TIFF file
     *
     * @param inFile livestack/sim.tif input file
     * @throws IOException
     * @throws InterruptedException
     */
    public void toMeta(String outFile_woExtension) throws IOException {

        for (int c = 0; c < header.channels.length; c++) {
            int count = 0;
            int lambda = header.channels[c].exWavelength;
            String outFile = outFile_woExtension + lambda + ".meta.txt";
            System.out.print("writing to " + outFile + " ...");
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
            System.out.print("...done");
        }
    }

    /**
     * ImageJ plugin to extract SIM sequences from stacks and their meta-data
     * file. Generate the meta-data file (and tiff stack, if required) first
     * with LiveSimConverter.
     */
    public static class LiveSimExtractor_ImageJplugin implements PlugIn {
        // parameters for sequence extraction

        static int minAvrIntensity;
        static long/*[] */ syncFrameDelay/* = {5000, 12995}*/;
        static long syncFrameDelayJitter;
        static int syncFrameInterval;
        static int simFramesPerSync;

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
            int countTimeFrame = 0, countAvrFrame = 0;

            // find sync frames
            for (int i = 0; i < allFrameList.size() - 10; i++) {

                MetaData md = allFrameList.get(i);

                long curTimeStamp = md.timeCam;

                // version 1 (for camera with precise time-stamp, like PCO)
                if (Math.abs(curTimeStamp - lastTimeStamp - syncFrameDelay) < syncFrameDelayJitter) {
                    md.isTimeSyncFrame = true;
                    syncFrameList.add(md);
                    IJ.log("found Syncframe " + i + " " + (curTimeStamp - lastTimeStamp - syncFrameDelay));
                    countTimeFrame++;
                }

                lastTimeStamp = curTimeStamp;

                // version 2 (for camera w/o timestamp, bright LED):
                if (md.average > minAvrIntensity) {
                    i++;
                    md = allFrameList.get(i);
                    md.isAvrSyncFrame = true;
                    syncFrameList.add(md);
                    countAvrFrame++;
                }
            }

            IJ.log("# sync frames (time): " + countTimeFrame);
            IJ.log("# sync frames (avr) : " + countAvrFrame);
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
                for (int i = simFramesPerSync; i < allFrameList.size() - simFramesPerSync; i += simFramesPerSync) {
                    MetaData fakeFrame = allFrameList.get(syncFrame.sortNr + i);
                    cleanSyncFrameList.add(fakeFrame);
                }
            } else {
                int discardedFrameCount = 0;

                MetaData lastEntry = syncFrameList.get(0);
                for (int i = 1; i < syncFrameList.size(); i++) {
                    MetaData curEntry = syncFrameList.get(i);
                    int distance = curEntry.sortNr - lastEntry.sortNr;

                    if (distance % syncFrameInterval == 0) {
                        cleanSyncFrameList.add(lastEntry);
                    } else {
                        discardedFrameCount++;
                    }

                    lastEntry = curEntry;
                    IJ.log("# Found " + cleanSyncFrameList.size() + " frames, discarded " + discardedFrameCount + " frames");
                }
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
                for (int j = 0; j < syncFrameInterval; j++) {
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

            String[] syncmethods = {"delay (PCO)", "brightness (hamamatsu)"};
            gd.addRadioButtonGroup("sync method", syncmethods, 2, 1, "delay (PCO)");

            String[] illuminationtime = {"1", "2", "5", "10"};
            gd.addRadioButtonGroup("illumination time", illuminationtime, 3, 1, "1");

            String[] brightness = {"5600", "10000", "18000", "32000", "56000"};
            gd.addRadioButtonGroup("syncframe brightness", brightness, 5, 1, "10000");

            gd.showDialog();
            if (gd.wasCanceled()) {
                System.out.println("gd canceled");
                return;
            }

            // ---- get parameters ----
            final String syncmethod = gd.getNextRadioButton();
            if (syncmethod.equals("delay (PCO)")) {

                final String tmp = gd.getNextRadioButton();
                switch (tmp) {
                    case "1": {
                        syncFrameDelay = 5000;
                        syncFrameDelayJitter = 16;
                        syncFrameInterval = 20;
                        simFramesPerSync = 18;
                        break;
                    }
                    case "2": {
                        syncFrameDelay = 5000;
                        syncFrameDelayJitter = 16;
                        syncFrameInterval = 20;
                        simFramesPerSync = 18;
                        break;
                    }
                    case "5": {
                        syncFrameDelay = 8000;
                        syncFrameDelayJitter = 16;
                        syncFrameInterval = 11;
                        simFramesPerSync = 9;
                        break;
                    }
                    case "10": {
                        syncFrameDelay = 12995;
                        syncFrameDelayJitter = 16;
                        syncFrameInterval = 11;
                        simFramesPerSync = 9;
                        break;
                    }
                }
                gd.getNextNumber();
                minAvrIntensity = 70000;
                System.out.println("test");
            } else if (syncmethod.equals("brightness (hamamatsu)")) {
                syncFrameDelay = 99999999;
                gd.getNextNumber();
                minAvrIntensity = (int) gd.getNextNumber();;
            } else {
                System.out.println("OOPS");
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
            printSyncFrameHistogramm();
            // clean up frame list
            cleanSyncFrameList();
            printReorderStats();
            // convert list
            List<Integer> simPositions = new ArrayList<Integer>();
            for (MetaData md : cleanSyncFrameList) {
                for (int i = 1; i <= simFramesPerSync; i++) {
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

    /**
     * for testing
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        boolean tif = false;
        boolean meta = false;
        String outFile;
        
        if(args.length>1)
        {
            if(args[0].compareToIgnoreCase("livesim2tif") == 0) tif = true;
            if(args[0].compareToIgnoreCase("livesim2meta") == 0) meta = true;
            if(args[0].compareToIgnoreCase("livesim2both") == 0) {
                tif = true;
                meta= true;
            }
        }
        if (args.length != 3 || (tif == false && meta == false)) {
            System.out.println("# Usage: Operation  Input-file  Output-folder");
            System.out.println("# where \"Operation\" is \"livesim2tif\", \"livesim2meta\" or \"livesim2both");
            System.exit(2);
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
        System.out.println("...done");
        if(tif) {
            outFile = outdir.getAbsolutePath() + file.separator + file.getName() + ".tif";
            System.out.println("\tsaving tif as: " + outFile + " ...");
            ls.saveAsTiff(outFile);
            System.out.println("...done");
        }
        if(meta) {
            outFile = outdir.getAbsolutePath() + file.separator + file.getName().replaceAll(".livesim$ || .livestack$","");
            System.out.println("\tsaving meta-file as: " + outFile + " ...");
            ls.toMeta(outFile);
            System.out.println("...done");
        }
        System.out.println("done");
    }
}
