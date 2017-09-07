/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fairsim.transport;

import ij.ImagePlus;
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
    
    enum FileFormat {
        TIFF, OMETIFF;
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
        final int nrPhases, nrAngles; // sim
        final int delayTime; // in ms
        final float samplePixelSize; // int nm
        final Channel[] channels;
        //final Conf.Folder xmlMetas;
        
        public Header(String microscope, String timestamp, String sample, String objective,
                int nrPhases, int nrAngles, int delayTime, float samplePixelSize, Channel[] channels) {
            
            this.microscope = microscope;
            this.timestamp = timestamp; //UTC timestamp in "yyyyMMdd'T'HHmmss" format
            this.sample = sample;
            this.objective = objective;
            this.nrPhases = nrPhases;
            this.nrAngles = nrAngles;
            this.delayTime = delayTime;
            this.samplePixelSize = samplePixelSize;
            this.channels = channels;
        }
        
        public static class Channel implements Serializable{
            
            static final long serialVersionUID = 1;
            final String detector, dye;
            final int width, height; // in pixels
            final int illuminationTime; // in µs
            final float illuminationPower; // in mW
            final int exWavelength; // wavelength of excitation in nm
            
            public Channel(String detector, String dye, int width, int height, int illuminationTime, float illuminationPower, int exWavelength) {
                
                this.detector = detector;
                this.dye = dye;
                this.width = width;
                this.height = height;
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
    
    ImageWrapper readImageWrapper(InputStream is) throws IOException {
        ByteBuffer bbHeader = ImageWrapper.readImageWrapperHeader(is);
        ImageWrapper iw = new ImageWrapper(bbHeader);
        //ImageWrapper iw = new ImageWrapper(header.channels[channelId].width, header.channels[channelId].height);
        //iw.readHeader(is);
        iw.readData(is);
        iw.parseHeader();
        return iw;
    }
    
    FileSaver saveAsTiff(String file, int channel) {
        FileSaver fs = new FileSaver(FileFormat.TIFF, file, channel);
        fs.start();
        return fs;
    }
    
    class FileSaver extends Thread {
        boolean preparing = false;
        boolean saving = false;
        boolean finished = false;
        int counter = 0;
        FileFormat format;
        String file;
        int wavelength;
        
        FileSaver(FileFormat format, String file, int wavelength) {
            this.format = format;
            this.file = file;
            this.wavelength = wavelength;
        }
        
        float getProgress() {
            if ((!preparing && !saving && !finished) || imgs.size() <= 0) return 0;
            else if (preparing) return (float) counter / imgs.size() * 5 / 10; // 8/10 after preperation
            else if (saving) return (float) 5 / 10; // 9/10 while writing on disk
            else if (finished) return 1;
            else throw new RuntimeException("Error in FileSaver progresss");
        }
        
        class IjStack extends ij.ImageStack {

            private IjStack(int width, int height) {
                super(width, height);
            }
            
            void addSlice(ImageWrapper iw) {
                ShortProcessor sp = new ShortProcessor(iw.width(), iw.height(), iw.getPixels(), null);
                super.addSlice(iw.encodeHeader(), sp);
            }
            
        }

        @Override
        public void run() {
            preparing = true;
            System.out.println("preparing");
            if (format == FileFormat.TIFF) {
                IjStack is = null;
                for (Header.Channel c : LiveStack.this.header.channels) {
                    if (wavelength == c.exWavelength) {
                        is = new IjStack(c.width, c.height);
                        break;
                    }
                }
                for (int i = 0; i < imgs.size(); i++) {

                    ImageWrapper iw = imgs.get(i);
                    if (iw.pos1() == wavelength) {
                        is.addSlice(iw);
                        
                    }
                    counter++;
                }
                try {
                    ImagePlus ip = new ImagePlus(header.encode(), is);
                    ij.io.FileSaver fs = new ij.io.FileSaver(ip);
                    saving = true;
                    System.out.println("saving");
                    preparing = false;
                    fs.saveAsTiffStack(file);
                    finished = true;
                    System.out.println("finished");
                    saving = false;
                } catch (IOException ex) {
                    Tool.error(ex.toString(), false);
                    ex.printStackTrace();
                }
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
//        Header.Channel c = new Header.Channel("t cam", "t dye", 512, 512, 1000, (float) 52.5, 525);
//        Header.Channel[] cs = {c};
//        Header h = new Header("t microscope", "t date", "t sample", "t objective",
//                3, 3, 250, (float) 79.2, cs);
//        ImageWrapper iw0 = new ImageWrapper(512, 512);
//        iw0.copy(new short[512 * 512], 512, 512);
//        ImageWrapper iw1 = new ImageWrapper(256, 256);
//        iw1.copy(new short[256 * 256], 256, 256);
//        
//        OutputStream fos = new FileOutputStream("D:/vigor-tmp/test.livestack");
//        
//        h.write(fos);
//        iw0.writeData(fos);
//        iw1.writeData(fos);
//        fos.close();
//        
//        InputStream is = new FileInputStream("D:/vigor-tmp/test.livestack");
//        LiveStack ls = new LiveStack(is);
        System.out.println("heap free: " + Runtime.getRuntime().freeMemory() / 1000000 + " mb");
        System.out.println("heap used: " + Runtime.getRuntime().totalMemory() / 1000000 + " mb");
        System.out.println("heap max: " + Runtime.getRuntime().maxMemory() / 1000000+ " mb");
        
        InputStream is = new FileInputStream("D:/vigor-tmp/VIGOR_20170907T140054.livestack");
        LiveStack ls = new LiveStack(is);
        FileSaver fs = ls.saveAsTiff("D:/vigor-tmp/livestack.tif", 488);
        while (!fs.finished) {
            Thread.sleep(100);
            System.out.println(fs.getProgress() * 100 + " %");
        }
    }
}
