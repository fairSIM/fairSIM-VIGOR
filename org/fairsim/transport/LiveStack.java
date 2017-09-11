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
import org.fairsim.utils.Conf;
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class LiveStack {
    
    private final Header header;
    private final List<ImageWrapper> imgs;
    
    private enum FileFormat {
        TIFF, OMETIFF;
    }
    
    LiveStack(InputStream is) throws IOException {
        header = Header.read(is);
        imgs = new LinkedList<>();
        while (is.available() > 0) {
            imgs.add(readImageWrapper(is));
        }
    }
    
    LiveStack(ImagePlus ip) throws IOException {
        String base64Header = ip.getInfoProperty();
        header = Header.decode(base64Header);
        imgs = new LinkedList<>();
        ImageStack stack = ip.getStack();
        for (int i = 1; i < stack.size() + 1; i++) {
            String info = stack.getSliceLabel(i);
            String encodedIWHeader = info.split("header: ")[1];
            ByteBuffer bbHeader = ImageWrapper.decodeHeader(encodedIWHeader);
            ImageWrapper iw = new ImageWrapper(bbHeader);
            short[] pixels = (short[]) stack.getPixels(i);
            iw.copy(pixels, iw.width(), iw.height());
            imgs.add(iw);
        }
    }
    
    public static LiveStack open(String file) throws IOException {
        LiveStack ls;
        if (file.endsWith(".livestack")) {
            FileInputStream fis = new FileInputStream(file);
            ls = new LiveStack(fis);
            fis.close();
        }
        else ls = new LiveStack(new Opener().openImage(file));
        return ls;
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
    
    FileSaverThread saveAsTiff(String outFile, boolean dump, int... channels) {
        FileSaverThread fc = new FileSaverThread(FileFormat.TIFF, outFile, dump, channels);
        fc.start();
        return fc;
    }
    
    FileSaverThread saveAsTiff(String outFile, int... channels) {
        return saveAsTiff(outFile, false, channels);
    }
    
    FileSaverThread saveAsTiff(String outFile) {
        return saveAsTiff(outFile, false);
    }
    
        
    public void sort() {
        imgs.sort(null);
    }
    
    private class FileSaverThread extends Thread {
        private String status = "starting";
        private int allCounter, addCounter;
        public final FileFormat format;
        private final String outFile;
        private final int[] channels;
        private final boolean dump;
        
        private FileSaverThread(FileFormat format, String outFile, boolean dump, int... channels) {
            this.format = format;
            this.outFile = outFile;
            this.dump = dump;
            
            int nrAllCh = header.channels.length;
            if (channels.length == 0) {
                this.channels = new int[nrAllCh];
                for (int i = 0; i < nrAllCh; i++) {
                    this.channels[i] = header.channels[i].exWavelength;
                }
            } else this.channels = channels;
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
        
        private void saveAsTiff() {
            status = "preparing";
            int nrCh = channels.length;
            sort();
            ImageStack is = new ImageStack(header.width, header.height);
            addCounter = 0;
            int size = imgs.size();
            for (allCounter = 0; allCounter < size; allCounter++) {
                ImageWrapper iw = dump ? imgs.remove(0) : imgs.get(allCounter);
                for (int c = 0; c < nrCh; c++) {
                    if (iw.pos1() == channels[c]) {
                        short[] pixels = iw.getPixels();
                        double avr=0;
                        for ( short p : pixels ) avr += p;
                        avr /= pixels.length;
                        // channel timeCapture avr seqNr HeaderBASE64
                        String sliceLabel = "ch: " + channels[c] +
                                " time: " + Tool.readableTimeStampMillis(iw.timeCapture() / 1000, true) +
                                " avr: " + avr +
                                " seqNr " + iw.seqNr() +
                                " header: " + iw.encodeHeader();
                        ShortProcessor sp = new ShortProcessor(iw.width(), iw.height(), pixels, null);
                        is.addSlice(sliceLabel, sp);
                        addCounter++;
                    }
                }
                if (dump && allCounter % 100 == 0) System.gc();
            }
            
            String info = "";
            try {
                info = header.encode();
            } catch (IOException ex) {
                throw new RuntimeException("this should never happen");
            }
            ImagePlus ip = new ImagePlus("", is);
            ip = HyperStackConverter.toHyperStack(ip, nrCh, header.zSlices, is.getSize()/nrCh/header.zSlices, "xyztc", "color");
            ip.setProperty("Info", info);
            ij.io.FileSaver fs = new ij.io.FileSaver(ip);
            status = "saving";
            fs.saveAsTiffStack(outFile);
            status = "finished";
        }
        
        @Override
        public void run() {
            if (format == FileFormat.TIFF) {
                saveAsTiff();
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
        /*
        long startTime = System.currentTimeMillis();
        InputStream is = new FileInputStream("D:/vigor-tmp/livestack-test_20170908T173839.livestack");
        System.out.println("read");
        LiveStack ls = new LiveStack(is);
        long tempTime = System.currentTimeMillis();
        long readTime = tempTime - startTime;
        System.out.println("save");
        FileSaverThread fc0 = ls.saveAsTiff("D:/vigor-tmp/livestack-asd.tif");
        while (fc0.isAlive()) {
        Thread.sleep(500);
        System.out.println(fc0.getAllCounter() + " ; " + fc0.getStatus());
        }
        long saveTime = System.currentTimeMillis() - tempTime;
        System.out.println("------------------");
        System.out.println("read: " + readTime);
        System.out.println("save: " + saveTime);
         */
        /*
        Opener opener = new Opener();
        ImagePlus testImg = opener.openImage("D:/vigor-tmp/livestack-asd.tif");
        String base64Header = testImg.getInfoProperty();
        Header header = Header.decode(base64Header);
        System.out.println(header.width);
        System.out.println(header.objective);
         */
        LiveStack ls = open("D:/vigor-tmp/livestack-asd.tif");

        
        ls.saveAsTiff("D:/vigor-tmp/livestack-488.tif", 488);
    }
}
