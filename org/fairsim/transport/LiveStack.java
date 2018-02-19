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
import ij.process.FloatProcessor;
import ij.process.ShortProcessor;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import org.fairsim.accel.AccelVectorFactory;
import org.fairsim.linalg.Vec;
import org.fairsim.linalg.Vec2d;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.livemode.ReconstructionRunner;
import org.fairsim.sim_algorithm.SimParam;
import org.fairsim.utils.Base64;
import org.fairsim.utils.Tool;
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
        sortAndFillupStack();
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
        sortAndFillupStack();
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
    /*
    private class LabeledImage {
        ImageWrapper iw;
        String label;
        
        LabeledImage(ImageWrapper pixels, String label) {
            this.iw = pixels;
            this.label = label;
        }
    }
    */
    /**
     * 
     */
    /*
    private List<LabeledImage> getLabeledImages() {
        orderImagesAsStack();
        int nrCh = header.channels.length;
        List<LabeledImage> labeledImgs = new ArrayList<>();
        for (int imgCounter = 0; imgCounter < imgs.size(); imgCounter++) {
            ImageWrapper iw = imgs.get(imgCounter);
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
                    LabeledImage li = new LabeledImage(iw, sliceLabel);
                    labeledImgs.add(li);
                }
            }
        }
        return labeledImgs;
    }
    */
    
    static String getSliceLabel(ImageWrapper iw) {
        short[] pixels = iw.getPixels();
        double avr = 0;
        for (short p : pixels) {
            avr += p;
        }
        avr /= pixels.length;
        // channel timeCapture avr seqNr HeaderBASE64
        String sliceLabel = "ch: " + iw.pos1()
                + " timestamp: " + iw.timeCamera()
                + " avr: " + avr
                + " seqNr " + iw.seqNr()
                + " header: " + iw.encodeHeader();
        return sliceLabel;
    }
    
    public ImagePlus convertToImagePlus(boolean dump) {
        sortAndFillupStack();
        ImageStack is = new ImageStack(header.width, header.height);
        int nrCh = header.channels.length;
        for (int imgCounter = 0; imgCounter < imgs.size(); imgCounter++) {
            ImageWrapper iw = dump ? imgs.remove(0) : imgs.get(imgCounter);
            for (int c = 0; c < nrCh; c++) {
                if (iw.pos1() == header.channels[c].exWavelength) {
                    /*
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
                    */
                    ShortProcessor sp = new ShortProcessor(iw.width(), iw.height(), iw.getPixels(), null);
                    String sliceLabel = getSliceLabel(iw);
                    is.addSlice(sliceLabel, sp);
                }
            }
        }
        /*
        int nrCh = header.channels.length;
        List<LabeledImage> labeledImages = getLabeledImages();
        ImageStack is = new ImageStack(header.width, header.height);
        for (LabeledImage li : labeledImages) {
            is.addSlice(li.label, li.iw.getPixels());
        }
        */
        ImagePlus ip = new ImagePlus("", is);
        ip = HyperStackConverter.toHyperStack(ip, nrCh, header.zSlices, is.getSize() / nrCh / header.zSlices, "xyztc", "color");
        ip.setProperty("Info", getHeaderString());
        return ip;
    }
    
    String getHeaderString() {
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
        for (int c = 0; c < header.channels.length; c++) {
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
        return info;
    }

    @Deprecated
    public int saveOmeTiff(String outFile, boolean dump) throws DependencyException, ServiceException, FormatException, IOException {
        int imgsPerChannel = sortAndFillupStack();
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
    private static VectorFactory getVectorFactory() {
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

    public ReconStack reconstructSim() {
        ReconstructionRunner.PerChannel[] pc = new ReconstructionRunner.PerChannel[header.channels.length];
        for (int i = 0; i < header.channels.length; i++) {      //get reconstructionParameters from LiveReconstruction
            if (header.channels[i].perChannel instanceof ReconstructionRunner.PerChannel) {
                pc[i] = (ReconstructionRunner.PerChannel) header.channels[i].perChannel;
            } else {
                throw new RuntimeException("need instance of ReconstructionRunner.PerChannel");
            }
        }
        SimReconstructor recRunner = new SimReconstructor(1, header.width, header.nrPhases, header.nrAngles, header.nrBands, pc);
        ImageWrapper[][][] shorts = getSimSequences();
        return recRunner.reconstruct(shorts);
    }
    
    public class ReconStack {
        
        String[][][] sliceLabel; // [t][c][pa]
        float[][][] widefield; //[t][c][xy]
        float[][][] recon; //[t][c][xy]
        
        ReconStack(String[][][] sliceLabel, float[][][] widefield, float[][][] recon) {
            this.sliceLabel = sliceLabel;
            this.widefield = widefield;
            this.recon = recon;
        }
        
        ImagePlus saveReconAsTiff(String outFile) {
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
                    String label = "";
                    for (int pa = 0; pa < nrPa; pa++) label += sliceLabel[t][c][pa];
                    is.addSlice(label, sp);
                }
            }
            ImagePlus ip = new ImagePlus("", is);
            ip = HyperStackConverter.toHyperStack(ip, nrCh, nrZ, is.getSize() / nrCh / nrZ, "xyzct", "color");
            ip.setProperty("Info", getHeaderString());
            ij.io.FileSaver fs = new ij.io.FileSaver(ip);
            fs.saveAsTiffStack(outFile);
            return ip;
        }
        
        ImagePlus saveWfAsTiff(String outFile) {
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
                    String label = "";
                    for (int pa = 0; pa < nrPa; pa++) label += sliceLabel[t][c][pa];
                    is.addSlice(label, sp);
                }
            }
            ImagePlus ip = new ImagePlus("", is);
            ip = HyperStackConverter.toHyperStack(ip, nrCh, nrZ, is.getSize() / nrCh / nrZ, "xyzct", "color");
            ip.setProperty("Info", getHeaderString());
            ij.io.FileSaver fs = new ij.io.FileSaver(ip);
            fs.saveAsTiffStack(outFile);
            return ip;
        }
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

        ReconStack reconstruct(ImageWrapper[][][] raws) {
            while (running || fitting) {
                sleeping(50); //wait for fit to finish
            }
            if (nrInReconQueue() > 0 || nrInFinalQueue() > 0) {
                Tool.error("LiveStack.Reconstructor: Queues should be empty");    //check if queues empty
            }
            int nrSimSeq = raws.length;
            int nrCh = raws[0].length;
            int nrPa = raws[0][0].length;
            int nrPixels = raws[0][0][0].width() * raws[0][0][0].height();
            String[][][] iwHeader = new String[nrSimSeq][nrCh][nrPa];
            float[][][] recon = new float[nrSimSeq][nrCh][nrPixels * 4];
            float[][][] widefield = new float[nrSimSeq][nrCh][nrPixels];
            running = true;
            Thread putThread = new Thread(new Runnable() {          //define new thread that pushes images from list "recons" to reconstruction
                public void run() {
                    for (int t = 0; t < nrSimSeq; t++) {
                        //ImageWrapper[][] iwArray = raws[t];  //extract next images for reconstruction
                        short[][][] raw = new short[nrCh][nrPa][];
                        //String[][] iwHeader = new String[lsHeader.channels.length][lsHeader.nrPhases * lsHeader.nrAngles];
                        for (int c = 0; c < nrCh; c++) {
                            for (int pa = 0; pa < nrPa; pa++) {
                                ImageWrapper iw = raws[t][c][pa];
                                //System.out.println(" " + nrSimSeq + " " + nrCh + " " + nrPa + " " + t + " " + c + " " + pa);
                                raw[c][pa] = iw.getPixels();
                                iwHeader[t][c][pa] = getSliceLabel(iw);
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
                            //float[][] hrFloats = new float[len][];
                            //float[][] wfFloats = new float[len][];
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
    
    ImageWrapper[][][] getSimSequences() {
        // get global stack information
        int iwPerChannel = sortAndFillupStack();
        int nrCh = header.channels.length;
        int nrPa = header.nrPhases * header.nrAngles;
        if (imgs.size() % nrCh != 0) throw new RuntimeException("Missmatch in iwPerChannel or nrCh " + imgs.size() + " " + nrCh);
        
        // split into channels
        List<List<ImageWrapper>> channelImgs = new ArrayList<>(nrCh);
        for (int c = 0; c < nrCh; c++) {
            List<ImageWrapper> channelList = new ArrayList<>();
            for (int i = 0; i < imgs.size(); i++) {
                ImageWrapper iw = imgs.get(i);
                if (iw.pos1() == header.channels[c].exWavelength) channelList.add(iw);
            }
            //System.out.println("channelList size: " + channelList.size());
            channelList.sort(null);
            List<ImageWrapper> cleanList = removeSyncs(channelList);
            channelImgs.add(cleanList);
        }
        iwPerChannel = channelImgs.get(0).size();
        for (List l : channelImgs) {
            if (l.size() != iwPerChannel) throw new RuntimeException("Missmatch in List cleanedImgs size" + l.size() + " " + iwPerChannel);
        }
        int nrTime = iwPerChannel / nrPa;
        
        /*
        // handle missing frames
        long[] firstSeqNr = new long[nrCh];
        for (int c = 0; c < nrCh; c++) firstSeqNr[c] = channelImgs.get(c).get(0).seqNr();
        for (int i = 0; i < iwPerChannel; i++) {
            for (int c = 0; c < nrCh; c++) {
                ImageWrapper iw1 = channelImgs.get(c).get(i);
                long count = iw1.seqNr() - firstSeqNr[c];
                if (count != i && iw1.seqNr() != Long.MAX_VALUE) 
                    throw new RuntimeException("missing frames in stack");
            }
        }
        
        //handle sync
        long[] last = new long[nrCh];
        List<List<Boolean>> sync = new ArrayList<>();
        for (int c = 0; c < nrCh; c++) {
            sync.add(new ArrayList<>());
            for (int i = 0; i < iwPerChannel; i++) {
                ImageWrapper iw = channelImgs.get(c).get(i);
                long current = iw.timeCamera();
                long diff = current - last[c];
                last[c] = current;
                if (diff > header.syncDelayTime) sync.get(c).add(true);
                else sync.get(c).add(false);
            }
        }
        */
        
        
        //System.out.println(nrTime + " " + nrCh + " " + nrPa + " " + iwPerChannel + " " + imgs.size()+ " " + channelImgs.size());
        ImageWrapper[][][] simSequences = new ImageWrapper[nrTime][nrCh][nrPa];
        for (int c = 0; c < nrCh; c++) {
            for (int i = 0; i < iwPerChannel; i++) {
                int pa = i % nrPa;
                int t = i  / nrPa;
                simSequences[t][c][pa] = channelImgs.get(c).get(i);
            }
        }
        /*
        for (int i = 0; i < channelImgs.size(); i++) {
            int pa = i % nrPa;
            int c = i / iwPerChannel;
            int t = (i % iwPerChannel) / nrPa;
            System.out.println("");
            //System.out.println("t c pa " + t + " " + c + " " + " " + pa);
            simSequences[t][c][pa] = channelImgs.get(c).get(i);
        }
        */
        return simSequences;
    }
    
    /*
    //andis preperation code prepares reconstruction [channels][p*a][pixels]
    private List<ImageWrapper[][]> extractSequences() {
        System.out.println("SYNCDELAY TIME IS " + header.syncDelayTime);
        List<List<ImageWrapper>> iwListList = new ArrayList<>();
        int outNr = imgs.size();
        System.out.println("header.channels.length == " + header.channels.length);
        if (header.channels.length == 0) {
            System.exit(10);
        }
        System.out.print("channels:");
        for (Header.Channel channel : header.channels) {
            System.out.print(" " + channel.exWavelength);
        }
        System.out.println("");
        for (Header.Channel channel : header.channels) {
            System.out.print("Channel " + channel.exWavelength);
            List<ImageWrapper> iwList = new ArrayList<>();
            for (ImageWrapper iw : imgs) {
                if (iw.pos1() == channel.exWavelength) {
                    iwList.add(iw);
                }
            }
            System.out.println(" Length " + iwList.size());
            iwList.sort(null);
            iwList = removeSyncs(iwList);
            iwListList.add(iwList);
            outNr = Math.min(outNr, iwList.size());
            System.out.println("\toutNr = " + outNr);
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
    */
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
        int nrSyncFrames = 0;

        //Get timestamps
        long[] timestamps = new long[nImgs];
        for (int i = 0; i < nImgs; i++) {
            timestamps[i] = outList.get(i).timeCamera();
        }

        //find syncframes
        System.out.println("\tfinding syncframes");
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
        System.out.println("\tremoving syncframes and known broken sim-sequences: ");
        reduce(outList, nonSimFrameList);
        System.out.println("\toutList.size = " + outList.size());
        nImgs = outList.size();

        //check sequence numbers
        System.out.println("\tchecking sequence-numbers of remaining images");
        List<Integer> brokenSeqNrList = checkSeqNr(outList, nrSimFrames);
        System.out.println("\tremoving newly found broken sim-sequences");
        reduce(outList, brokenSeqNrList);
        System.out.println("\toutList.size = " + outList.size());

        return outList;
    }

    private List<Integer> findSyncFrames(List<ImageWrapper> iwList, long[] timestamps, int syncFrameDelay, int syncFrameDelayJitter) {
        int nImgs = iwList.size();
        List<Integer> syncFrameList = new ArrayList<>();
//        for (int i = 1; i < nImgs; i++) {
//            System.out.println("delay "+ (Math.abs(timestamps[i] - timestamps[i - 1])));
//        }
        System.out.print("\t\tfound Syncframes: ");
        syncFrameList.add(0);
        System.out.print("0, ");
        for (int i = 1; i < nImgs; i++) {
            if (timestamps[i] - timestamps[i - 1] > syncFrameDelay) {
                syncFrameList.add(i);
                System.out.print(i + ", ");
            }
        }
        System.out.println("done");
        if (syncFrameList.isEmpty()) {
            System.err.println("\t\tNo Sync-frames found");
            System.exit(1);
        }
        return syncFrameList;
    }

    private List<Integer> findNonSimFrames(List<ImageWrapper> iwList, List<Integer> syncFrameList, int nrSimFrames, int nrSyncFrames) {
        System.out.print("\t\tfinding non-SIM-frames... ");
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

    private List<Integer> checkSeqNr(List<ImageWrapper> iwList, int nrSimFrames) {
        System.out.println("\t\tchecking seq Nr. Mismatches: ");
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
        System.out.println("\t\tdone");
        return brokenSeqNrList;
    }

    private /*List<ImageWrapper>*/ void reduce(List<ImageWrapper> inList, List<Integer> brokenSeqNr) {
        if (brokenSeqNr.isEmpty()) {
            System.out.println("\t\t\tnothing to remove");
//            return inList;
        }
//        List<ImageWrapper> outList = new ArrayList<>();
//        for (int i = 0; i < inList.size(); i++) {
//            outList.add(inList.get(i));
//        }
        Collections.sort(brokenSeqNr);
        System.out.println("\t\t\tremoving " + brokenSeqNr.size() + " frames from list with length " + inList.size() + ": ");
        for (int i = brokenSeqNr.size() - 1; i >= 0; i--) {
//            System.out.print(red.get(i)+", ");
            inList.remove((int) brokenSeqNr.get(i));
        }
        System.out.println("\t\tdone, new length: " + inList.size());
        System.out.println("\tdone, new length: " + inList.size());
//        return outList;
    }

    /**
     * creates livesim-style metadata from a livestack/livesim-TIFF file
     *
     * @param outFile_woExtension
     * @throws IOException
     */
    public void toMeta(String outFile_woExtension) throws IOException {

        for (int c = 0; c < header.channels.length; c++) {
            int count = 0;
            int lambda = header.channels[c].exWavelength;
            String outFile = outFile_woExtension + lambda + ".meta.txt";
            System.out.print("\t\twriting to " + outFile + " ... ");
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
            System.out.println(" writing done");
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
            
            ReconStack reconStack = ls.reconstructSim();
            reconStack.saveReconAsTiff(reconFile);
            reconStack.saveWfAsTiff(wfFile);
            
//            float[][][] fl;
//            fl = ls.reconstructSim().recon;
//
//            ImageStack is = new ImageStack(2 * ls.header.width, 2 * ls.header.height);
//            //int nrCh = ls.header.channels.length;
//            //float[] pixels = new float[2 * ls.header.width * 2 * ls.header.height];
//            for (int imgCounter = 0; imgCounter < fl.length; imgCounter++) {
//                //System.out.println("fl.size() " + fl.size() + "\tfl.get().length " + fl.get(imgCounter).length + "\tfl.get()[].length " + fl.get(imgCounter)[0].length);
//                float[][] channelImages = fl[imgCounter];
//                for (int ch = 0; ch < channelImages.length; ch++) {
//                    //System.out.println("ch = " + ch);
//                    FloatProcessor fpp = new FloatProcessor(2 * ls.header.width, 2 * ls.header.width, channelImages[ch]);
//                    is.addSlice("test channel " + ch, fpp);
//                }
//                /*
//                for (int j = 0; j < (2 * ls.header.width * 2 * ls.header.height); j++) {
//                    System.out.println("\tpixels["+j+"] = fl.get("+imgCounter+")["+j / (2 * ls.header.width)+"]["+j % (2 * ls.header.height)+"];");
//                    pixels[j] = fl.get(imgCounter)[j % 2 * ls.header.width][j / 2 * ls.header.height];
//                }
//                FloatProcessor fp;
//                fp = new FloatProcessor(2 * ls.header.width, 2 * ls.header.width, pixels, null);
//                is.addSlice("test channel " + nrCh, fp);
//                */
//            }
//
//            System.out.println("\tdone\n");
//
//            //this saves the original file... OOPS
//            System.out.print("\tsaving tif as: " + outFile + " ...");
//            ImagePlus ip = new ImagePlus("Test Title" ,is);
//            ij.io.FileSaver fs = new ij.io.FileSaver(ip);
//            fs.saveAsTiffStack(outFile);
            
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
                ls.saveAsTiff(outFile);
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
