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

package org.fairsim.transport;

import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.HyperStackConverter;
import ij.process.ShortProcessor;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
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
import org.fairsim.transport.ImageWrapper.Sorter.SorterException;
import org.fairsim.utils.Base64;
import org.fairsim.utils.Conf;
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class LiveStack {
    
    private Header header;
    List<ImageWrapper> imgs;
    
    private enum FileFormat {
        TIFF, OMETIFF;
    }
    
    private enum Mode {
        CONVERTER, SAVER;
    }
    
    LiveStack(InputStream is) throws IOException {
        header = Header.read(is);
        imgs = new LinkedList<>();
        while (is.available() > 0) {
            imgs.add(readImageWrapper(is));
        }
    }
    
    LiveStack(File file) throws IOException {
        this(new FileInputStream(file));
    }
    
    public static class Header implements Serializable{
        
        static final long serialVersionUID = 1;
        final String microscope, timestamp, sample, objective;
        final int width, height; // in pixels
        final int zSlices;
        final int nrPhases, nrAngles; // sim
        final int delayTime; // in ms
        final float samplePixelSize; // int nm
        final Channel[] channels;
        //final Conf.Folder xmlMetas;
        
        public Header(String microscope, String timestamp, String sample,
                String objective, int width, int height, int zSlices,
                int nrPhases, int nrAngles, int delayTime,
                float samplePixelSize, Channel[] channels) {
            
            this.microscope = microscope;
            this.timestamp = timestamp; //UTC timestamp in "yyyyMMdd'T'HHmmss" format
            this.sample = sample;
            this.objective = objective;
            this.width = width;
            this.height = height;
            this.zSlices = zSlices;
            this.nrPhases = nrPhases;
            this.nrAngles = nrAngles;
            this.delayTime = delayTime;
            this.samplePixelSize = samplePixelSize;
            this.channels = channels;
        }
        
        public static class Channel implements Serializable{
            
            static final long serialVersionUID = 1;
            final String detector, dye;
            final int illuminationTime; // in µs
            final float illuminationPower; // in mW
            final int exWavelength; // wavelength of excitation in nm
            
            public Channel(String detector, String dye, int illuminationTime, float illuminationPower, int exWavelength) {
                
                this.detector = detector;
                this.dye = dye;
                this.illuminationTime = illuminationTime;
                this.illuminationPower = illuminationPower;
                this.exWavelength = exWavelength;
            }
            
        }
        
        void write(OutputStream os) throws IOException {
            ObjectOutputStream oos = new ObjectOutputStream(os);
            oos.writeObject(this);
            oos.flush();
        }
        
        static Header read(InputStream is) throws IOException {
            ObjectInputStream ois = new ObjectInputStream(is);
            try {
                return (Header) ois.readObject();
            } catch (ClassNotFoundException ex) {
                throw new IOException(ex);
            }
        }
        
        String encode() throws IOException {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            write(baos);
            baos.close();
            byte[] bytes = baos.toByteArray();
            return Base64.encode(bytes);
        }
        
        static Header decode(String encoded) throws IOException {
            byte[] bytes = Base64.decode(encoded);
            ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
            return read(bais);
        }
    }
    
    void writeHeader(OutputStream os) throws IOException {
        header.write(os);
    }
    
    static ImageWrapper readImageWrapper(InputStream is) throws IOException {
        ByteBuffer bbHeader = ImageWrapper.readImageWrapperHeader(is);
        ImageWrapper iw = new ImageWrapper(bbHeader);
        iw.readData(is);
        iw.parseHeader();
        return iw;
    }
    
    FileConverter saveAsTiff(String outFile, int... channels) {
        FileConverter fc = new FileConverter(FileFormat.TIFF, this, outFile, channels);
        fc.start();
        return fc;
    }
    
    FileConverter saveAsTiff(String outFile) {
        FileConverter fc = new FileConverter(FileFormat.TIFF, this, outFile);
        fc.start();
        return fc;
    }
    
    static FileConverter liveStackToTiff(String inFile, String outFile, int... channels) throws IOException {
        FileConverter fc = new FileConverter(FileFormat.TIFF, inFile, outFile, channels);
        fc.start();
        return fc;
    }
    
    static FileConverter liveStackToTiff(String inFile, String outFile) throws IOException {
        FileConverter fc = new FileConverter(FileFormat.TIFF, inFile, outFile);
        fc.start();
        return fc;
    }
    
    static private class FileConverter extends Thread {
        private String status = "starting";
        private int allCounter = 0, addCounter = 0;
        private Mode mode;
        public final FileFormat format;
        private LiveStack ls;
        private Header header;
        private FileInputStream inStr;
        private final String outFile;
        private int[] channels;
        private ImageWrapper.Sorter[] sorter;
        
        private final void init(int[] channels) {
            int nrAllCh = header.channels.length;
            
            if (channels.length == 0) {
                this.channels = new int[nrAllCh];
                for (int i = 0; i < nrAllCh; i++) {
                    this.channels[i] = header.channels[i].exWavelength;
                }
            } else this.channels = channels;
            
            int nrCh = this.channels.length;
            sorter = new ImageWrapper.Sorter[nrCh];
            for (int i = 0; i < nrCh; i++) {
                sorter[i] = new ImageWrapper.Sorter();
            }
        }
        
        private FileConverter(FileFormat format, String inFile, String outFile, int... channels) throws IOException {
            this.format = format;
            this.outFile = outFile;
            mode = Mode.CONVERTER;
            inStr = new FileInputStream(inFile);
            header = Header.read(inStr);
            init(channels);
        }
        
        private FileConverter(FileFormat format, LiveStack ls, String outFile, int... channels) {
            this.format = format;
            this.outFile = outFile;
            mode = Mode.SAVER;
            this.ls = ls;
            header = ls.header;
            init(channels);
        }

        public String getStatus() {
            return status;
        }
        
        public int getAllCounter() {
            return allCounter;
        }
        
        public int getAddCounter() {
            return addCounter;
        }
        
        private boolean hasNextImg() throws IOException {
            if (mode == Mode.CONVERTER) return inStr.available() > 0;
            else if (mode == Mode.SAVER) return ls.imgs.size() > allCounter;
            else throw new IOException("Invalid mode");
        }
        
        private ImageWrapper getNextImg() throws IOException {
            if (mode == Mode.CONVERTER) {
                allCounter++;
                return readImageWrapper(inStr);
            }
            else if (mode == Mode.SAVER) return ls.imgs.get(allCounter++);
            else throw new IOException("Invalid mode");
            
        }

        private void addToSorter(ImageWrapper iw) throws SorterException {
            for (int c = 0; c < channels.length; c++) {
                if (iw.pos1() == channels[c]) {
                    sorter[c].add(iw);
                }
            }
        }
        
        private ImageWrapper getFromSorter(int channelIdx) throws IOException, SorterException { 
            ImageWrapper iw;
            try {
                iw = sorter[channelIdx].poll();
            } catch (SorterException ex) {
                if (hasNextImg()) {
                    addToSorter(getNextImg());
                    iw = getFromSorter(channelIdx);
                } else throw ex;
            }
            if (iw == null) {
                if (hasNextImg()) {
                    addToSorter(getNextImg());
                    iw = getFromSorter(channelIdx);
                }
            }
            return iw;
        }

        
        private void convertToTiff() {
            int nrCh = channels.length;
            status = "preparing";
            
            try {
                ImageStack is = new ImageStack(header.width, header.height);
                for (int i = 0; i < 11; i++) {
                    if (hasNextImg()) {
                        ImageWrapper iw = getNextImg();
                        addToSorter(iw);
                    }
                }
                boolean loop = true;
                while (loop) {
                    
                    for (int c = 0; c < nrCh; c++) {
                        ImageWrapper iw = getFromSorter(c);
                        if (iw == null) {
                            loop = false;
                            break;
                        }
                        if (iw.pos1() == channels[c]) {
                            ShortProcessor sp = new ShortProcessor(iw.width(), iw.height(), iw.getPixels(), null);
                            is.addSlice("rec: " + Tool.readableTimeStampMillis(iw.timeCapture()/1000,true)+ " ch: " + channels[c] + " header: "+iw.encodeHeader(), sp);
                            addCounter++;
                        }
                    }
                }
                
                String isHeader = header.encode();
                ImagePlus ip = new ImagePlus(isHeader, is);
                ip = HyperStackConverter.toHyperStack(ip, nrCh, header.zSlices, is.getSize()/nrCh, "default", "color");
                ij.io.FileSaver fs = new ij.io.FileSaver(ip);
                status = "saving";
                fs.saveAsTiffStack(outFile);
            } catch (IOException | ImageWrapper.Sorter.SorterException ex) {
                Tool.error(ex.toString(), false);
                ex.printStackTrace();
            }
            status = "finished";
        }
        
        @Override
        public void run() {
            if (format == FileFormat.TIFF) {
                convertToTiff();
            } else if (format == FileFormat.OMETIFF) {
                //TODO
            }
        }
    }
    
    
    /**
     * for testing
     * @param args
     * @throws FileNotFoundException
     * @throws IOException
     * @throws ClassNotFoundException 
     */
    public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException, Conf.EntryNotFoundException, InterruptedException {
        
        long currentTimeMillis = System.currentTimeMillis();
        InputStream is = new FileInputStream("D:/vigor-tmp/livestack-test_20170908T173839.livestack");
        LiveStack ls = new LiveStack(is);
        FileConverter fc0 = ls.saveAsTiff("D:/vigor-tmp/livestack.tif");
        while (fc0.isAlive()) {
            Thread.sleep(500);
            System.out.println(fc0.getAllCounter() + " ; " + fc0.getStatus());
        }
        long saveTime = System.currentTimeMillis() - currentTimeMillis;
        
        currentTimeMillis = System.currentTimeMillis();
        FileConverter fc1 = liveStackToTiff("D:/vigor-tmp/livestack-test_20170908T173839.livestack", "D:/vigor-tmp/livestack_stream.tif");
        while (fc1.isAlive()) {
            Thread.sleep(500);
            System.out.println(fc1.getAllCounter() + " ; " + fc1.getStatus());
        }
        long convertTime = System.currentTimeMillis() - currentTimeMillis;
        
        System.out.println("------------------");
        System.out.println("save: " + saveTime);
        System.out.println("convert: " + convertTime);
    }
}
