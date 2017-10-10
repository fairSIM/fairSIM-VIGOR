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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;
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
import loci.formats.ImageReader;

import ome.xml.model.enums.DimensionOrder;
import ome.xml.model.enums.PixelType;
import ome.xml.model.enums.*;
import ome.xml.model.primitives.*;

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

    /**
     * private constructor for livestack InputStreams use the open method for
     * creating a livestack instance
     *
     * @param is InputStream for the livestack instance
     * @throws IOException
     */
    private LiveStack(InputStream is) throws IOException {
        header = Header.read(is);
        imgs = new LinkedList<>();
        while (is.available() > 0) {
            imgs.add(readImageWrapper(is));
        }
    }

    /**
     * private constructor for ImagePlus instance of ImageJ
     *
     * @param ip livestack imageplus
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
    }

    /**
     * creates a livestack instance from a *.livestack or a *.livestack.* file
     *
     * @param file absolute path of the file
     * @return livestack instance of the file
     * @throws IOException
     */
    public static LiveStack open(String file) throws IOException {
        System.out.println("livestack_open");
        LiveStack ls;
        if (file.endsWith(".livestack")) {
            FileInputStream fis = new FileInputStream(file);
            ls = new LiveStack(fis);
            fis.close();
        } else {
            ls = new LiveStack(new Opener().openImage(file));
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
        final int nrPhases, nrAngles; // sim
        final int illuminationTime; // in µs
        final int delayTime; // in ms
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
         * @param nrAngles amoumt of SIM angles
         * @param illuminationTime in µs
         * @param delayTime in ms
         * @param samplePixelSizeX in nm
         * @param samplePixelSizeY in nm
         * @param samplePixelSizeZ in nm
         * @param channels channel meta data
         */
        public Header(String microscope, String timestamp, String sample,
                String objective, int width, int height, int zSlices,
                int nrPhases, int nrAngles, int illuminationTime, int delayTime,
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
            this.illuminationTime = illuminationTime;
            this.delayTime = delayTime;
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

            /**
             *
             * @param detector
             * @param dye
             * @param illuminationPower in mW
             * @param exWavelength in nm
             */
            public Channel(String detector, String dye, float illuminationPower, int exWavelength) {

                this.detector = detector;
                this.dye = dye;
                this.illuminationPower = illuminationPower;
                this.exWavelength = exWavelength;
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
            System.out.println("livestack_run");
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
                                + " time: " + Tool.readableTimeStampMillis(iw.timeCapture() / 1000, true)
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

            } catch (Exception ex) {
                throw new RuntimeException("OOOPS");
            }
        }

        /**
         * saves this livestack instance as file
         */
        @Override
        public void run() {
            preparation();
            if (omeTiff) saveOmeTiff();
            if (tiff) saveTiff();
        }
    }
    
    public static void convert(String inFile, boolean saveOmeTiff, boolean saveTiff) {
        // hier weiter machen
    }
    
    /**
     * for testing
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        liveStackToTiff("/home/andi/git/fairSIM/fastSIM_20170919T140051.livestack");
        
        LiveStack ls = open("/home/andi/git/fairSIM/fastSIM_20170919T140051.livestack");
    }

}
