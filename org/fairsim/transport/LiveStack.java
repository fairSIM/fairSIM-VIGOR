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
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class LiveStack {
    
    private final Header header;
    private final List<ImageWrapper> imgs;
    
    LiveStack(InputStream is) throws IOException {
        header = Header.read(is);
        imgs = new LinkedList<>();
        while (is.available() > 0) {
            imgs.add(readImageWrapper(is));
        }
    }
    
    LiveStack(ImagePlus ip) throws IOException {
        String base64Header = ip.getInfoProperty().split("encoded header: ")[1];
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
        final int illuminationTime; // in µs
        final int delayTime; // in ms
        final float samplePixelSizeX, samplePixelSizeY, samplePixelSizeZ; // int nm
        final Channel[] channels;
        
        public Header(String microscope, String timestamp, String sample,
                String objective, int width, int height, int zSlices,
                int nrPhases, int nrAngles, int illuminationTime, int delayTime,
                float samplePixelSizeX, float samplePixelSizeY, Channel[] channels) {
            
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
            this.samplePixelSizeZ = 0;
            this.channels = channels;
        }
        
        public static class Channel implements Serializable{
            
            static final long serialVersionUID = 1;
            final String detector, dye;
            final float illuminationPower; // in mW
            final int exWavelength; // wavelength of excitation in nm
            
            public Channel(String detector, String dye, float illuminationPower, int exWavelength) {
                
                this.detector = detector;
                this.dye = dye;
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
    
    private void writeHeader(OutputStream os) throws IOException {
        header.write(os);
    }
    
    public static ImageWrapper readImageWrapper(InputStream is) throws IOException {
        ByteBuffer bbHeader = ImageWrapper.readImageWrapperHeader(is);
        ImageWrapper iw = new ImageWrapper(bbHeader);
        iw.readData(is);
        iw.parseHeader();
        return iw;
    }
    
    public FileSaverThread saveAsTiff(String outFile, boolean dump, int... channels) {
        FileSaverThread fc = new FileSaverThread(outFile, dump, channels);
        fc.start();
        return fc;
    }
    
    public FileSaverThread saveAsTiff(String outFile, int... channels) {
        return saveAsTiff(outFile, false, channels);
    }
    
    public static void liveStacktoTiff(String inFile, int... channels) throws IOException, InterruptedException {
        LiveStack ls = open(inFile);
        String outFile = inFile + ".tif";
        FileSaverThread fst = ls.saveAsTiff(outFile, channels);
        fst.join();
    }
    
    private class FileSaverThread extends Thread {
        private String status = "starting";
        private int allCounter, addCounter;
        private final String outFile;
        private final int[] channels;
        private final boolean dump;
        
        private FileSaverThread(String outFile, boolean dump, int... channels) {
            this.outFile = outFile;
            this.dump = dump;
            
            int nrAllCh = header.channels.length;
            if (channels.length == 0) {
                this.channels = new int[nrAllCh];
                for (int i = 0; i < nrAllCh; i++) {
                    this.channels[i] = header.channels[i].exWavelength;
                }
            } else this.channels = channels;
            for (int channel : this.channels) {
                boolean found = false;
                for (Header.Channel hc : header.channels) {
                    if (channel == hc.exWavelength) {
                        found = true;
                        break;
                    }
                }
                if(!found) throw new RuntimeException("channel " + channel + " not found");
            }
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
        
        @Override
        public void run() {
            status = "preparing";
            int nrCh = channels.length;
            int[] chCounter = new int[nrCh];
            for (ImageWrapper iw : imgs) {
                for (int c = 0; c < nrCh; c++) {
                    if (iw.pos1() == header.channels[c].exWavelength) {
                        chCounter[c]++;
                    }
                }
            }
            int max = 0;
            for (int c : chCounter) {
                if (c > max) {
                    max = c;
                }
            }
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
            ImageStack is = new ImageStack(header.width, header.height);
            addCounter = 0;
            for (allCounter = 0; allCounter < imgs.size(); allCounter++) {
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
                        //System.out.println(ImageWrapper.decodeHeader(iw.encodeHeader()).get(17));
                        ShortProcessor sp = new ShortProcessor(iw.width(), iw.height(), pixels, null);
                        is.addSlice(sliceLabel, sp);
                        addCounter++;
                    }
                }
                if (dump && allCounter % 100 == 0) System.gc();
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
            info += "illuminationTime: " + header.illuminationTime + " ms \n";
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
            ip = HyperStackConverter.toHyperStack(ip, nrCh, header.zSlices, is.getSize()/nrCh/header.zSlices, "xyztc", "color");
            ip.setProperty("Info", info);
            ij.io.FileSaver fs = new ij.io.FileSaver(ip);
            status = "saving";
            fs.saveAsTiffStack(outFile);
            status = "finished";
        }
    }
    
    
    /**
     * for testing
     * @param args
     * @throws FileNotFoundException
     * @throws IOException
     * @throws ClassNotFoundException 
     */
    public static void main(String[] args) throws Exception {
        liveStacktoTiff("D:/vigor-tmp/fastSIM_20170919T140051.livestack");
        LiveStack open = open("D:/vigor-tmp/fastSIM_20170919T140051.tif");
    }
}
