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

package org.fairsim.registration;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.DataFormatException;
import org.fairsim.linalg.*;
import org.fairsim.utils.Conf;
import org.fairsim.utils.Tool;

/**
 * Class to register images to each other
 * @author m.lachetta
 */
public class Registration {

    private final String channelName;
    private static final VectorFactory vf = Vec.getBasicVectorFactory();
    private int reconMaxW;
    private int reconMaxH;
    private int wfMaxW;
    private int wfMaxH;
    private final Vec2d.Real reconXTransVec;
    private final Vec2d.Real reconYTransVec;
    private final Vec2d.Real wfXTransVec;
    private final Vec2d.Real wfYTransVec;
    static private int threads;
    private static final List<Registration> registrations = new ArrayList<>();
    private static boolean widefield;
    private static boolean recon;
    
    /**
     * Constructor for a registration object
     * @param file raw registration file of bUnwrappedJ
     * @throws IOException is thrown if the input file does not exist or not
     * have the correct structure/format
     */
    private Registration(String file, String channelName) throws IOException {
        this.channelName = channelName;
        reconMaxW = -1;
        reconMaxH = -1;
        wfMaxW = -1;
        wfMaxH = -1;

        // read in the registration file
        BufferedReader br = new BufferedReader(new FileReader(file));
        Tool.trace("Registration: readin file: " + file);

        reconMaxW = Integer.parseInt(br.readLine().split("=")[1]);
        reconMaxH = Integer.parseInt(br.readLine().split("=")[1]);
        wfMaxW = reconMaxW / 2;
        wfMaxH = reconMaxH / 2;

        reconXTransVec = vf.createReal2D(reconMaxW, reconMaxH);
        reconYTransVec = vf.createReal2D(reconMaxW, reconMaxH);
        wfXTransVec = vf.createReal2D(wfMaxW, wfMaxH);
        wfYTransVec = vf.createReal2D(wfMaxW, wfMaxH);
        readInTransVector(br);

        br.close();
        Tool.trace("Registration: readin done for: " + file);

        // sets number of threads for registering images
        threads = Runtime.getRuntime().availableProcessors();
        if (threads > wfMaxH) {
            threads = wfMaxH;
        }
        if (threads < 1) {
            threads = 1;
        }
    }

    /**
     * Creates a new Registration and adds it to the list of registrations
     * @param regFolder folder for registration files
     * @param channelName channel (wavelength) of the registration
     * @return true if registration could be created, else false
     */
    public static boolean createRegistration(String regFolder, String channelName) {
        try {
            String filename = Tool.getFile(regFolder + channelName + ".txt").getAbsolutePath();
            Registration reg = new Registration(filename, channelName);
            registrations.add(reg);
            Tool.trace("Registration: Registrering channel: " + channelName);
            return true;
        } catch (IOException ex) {
            Tool.trace("Registration: No registration for channel: " + channelName);
            return false;
        }
    }
    
    /**
     * Removes the registration for a specified channel
     * @param channelName channel for which the registration should be removed
     */
    public static void clearRegistration(String channelName) {
        try {
            Registration reg = getRegistration(channelName);
            int channelId = reg.getRegId();
            registrations.remove(channelId);
        } catch (NoSuchFieldException ex) {}
    }
    
    /**
     * 
     * @return the id of this registration
     * @throws NoSuchFieldException if this registration is not in the registration list
     */
    private int getRegId() throws NoSuchFieldException {
        Registration reg;
        for (int i = 0; i < registrations.size(); i++) {
            reg = registrations.get(i);
            if (reg.channelName.equals(this.channelName)) {
                return i;
            }
        }
        throw new NoSuchFieldException("There is no registration for: " + channelName);
    }
    
    /**
     * Reads the directory for registrations from an Conf.Folder if no registration
     * folder is set 
     * @param cfg Conf.Folder for read in
     * @return directory for registrations
     * @throws FileNotFoundException if the registration folder does not exists
     */
    public static String getRegFolder(final Conf.Folder cfg) throws FileNotFoundException {
        String regFolder;// = "(not found)";
        try {
            regFolder = Tool.getFile(cfg.getStr("RegistrationFolder").val()).getAbsolutePath();
        } catch (Conf.EntryNotFoundException ex) {
            regFolder = System.getProperty("user.dir");
            Tool.error("Registration: No registration folder found. Registration Folder was set to: " + regFolder, false);
        }
        File file = new File(regFolder);
        if (file.exists()) {
            return regFolder;
        } else {
            throw new FileNotFoundException("Registration folder does not exists");
        }
    }
    
    /**
     * Finds the required registration of the specific channel
     * @param channelName channel of the required registration
     * @return registration object for a specific channel
     * @throws NoSuchFieldException is thrown if the required registration
     * does not exists
     */
    public static Registration getRegistration(String channelName) throws NoSuchFieldException {
        for (Registration reg : registrations) {
            if (reg.channelName.equals(channelName)) {
                return reg;
            }
        }
        throw new NoSuchFieldException("There is no registration for: " + channelName);
    }
    
    /**
     * (de)activates the registration in wide field
     * @param b 
     */
    public static void setWidefield(boolean b) {
        widefield = b;
    }
    
    /**
     * (de)activates the registration in reconstruction
     * @param b 
     */
    public static void setRecon(boolean b) {
        recon = b;
    }
    
    /**
     * 
     * @return is registration in wide field active?
     */
    public static boolean isWidefield() {
        return widefield;
    }
    
    /**
     * 
     * @return is registration in wide field active?
     */
    public static boolean isRecon() {
        return recon;
    }
    
    /**
     * Registers a wide field image
     * @param sourceVec input vector for the registration
     * @return output vector from the registration
     */
    public Vec2d.Real registerWfImage(Vec2d.Real sourceVec) throws DataFormatException {
        int width = sourceVec.vectorWidth();
        int height = sourceVec.vectorHeight();
        if (width > wfMaxW || height > wfMaxH) {
            throw new DataFormatException("Image registration failed, because of differences in diminsions");
        }
        return registerImage(sourceVec, wfXTransVec, wfYTransVec, width, height);
    }
    
    /**
     * Registers a reconstructed vector
     * @param sourceVec input vector for the registration
     * @return output vector from the registration
     */
    public Vec2d.Real registerReconImage(Vec2d.Real sourceVec) throws DataFormatException {
        int width = sourceVec.vectorWidth();
        int height = sourceVec.vectorHeight();
        if (width > reconMaxW || height > reconMaxH) {
            throw new DataFormatException("Image registration failed, because of differences in diminsions");
        }
        return registerImage(sourceVec, reconXTransVec, reconYTransVec, width, height);
    }
    
    /**
     * Registers a vector
     * @param sourceVec input vector for the registration
     * @param xTransVec transformation matrix in X
     * @param yTransVec transformation matrix in Y
     * @param width width of the vector
     * @param height height of the vector
     * @return output vector from the registration
     */
    Vec2d.Real registerImage(final Vec2d.Real sourceVec, 
	final Vec2d.Real xTransVec, final Vec2d.Real yTransVec, 
	final int width, final int height) {
        
        // preperation for the registration
        final Vec2d.Real regVec = vf.createReal2D(width, height);
        int maxWidth = xTransVec.vectorWidth();
        int maxHeight = xTransVec.vectorHeight();
        if (maxWidth != yTransVec.vectorWidth() || maxHeight != yTransVec.vectorHeight()) {
            throw new RuntimeException("missmatch in dimensions");
        }
        int offsetX = (maxWidth - width) / 2;
        int offsetY = (maxHeight - height) / 2;
        
        //registers an image in a multi threaded way
        final int blockSize = height / threads;
        final Thread[] blocks = new Thread[threads];
        for (int threadId = 0; threadId < threads; threadId++) {
            final int tId = threadId;
            blocks[tId] = new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int y = blockSize*tId + offsetY; y < blockSize*(tId + 1) + offsetY; y++) {
                        for (int x = offsetX; x < width + offsetX; x++) {
                            transPixel(regVec, sourceVec, xTransVec, yTransVec, width, height, x, y, offsetX, offsetY);
                        }
                    }
                }
            });
            blocks[tId].start();
        }

        for (int y = blockSize * threads; y < height; y++) {
            for (int x = 0; x < width; x++) {
                transPixel(regVec, sourceVec, xTransVec, yTransVec, width, height, x, y, offsetX, offsetY);
            }
        } 
        
        for (int threadId = 0; threadId < threads; threadId++) {
            try {
                blocks[threadId].join();
            } catch (InterruptedException ex) {
                Tool.error("Registration thread interrupted!", false);
            }
        }
        
        return regVec;
    }
    
    /**
     * Registers a single pixel in the output vector from the registration
     * @param regVec output vector from the registration
     * @param sourceVec input vector for the registration
     * @param xTransVec transformation matrix in X
     * @param yTransVec transformation matrix in Y
     * @param width width of the Vector
     * @param height height of the Vector
     * @param targetX x value of the target pixel
     * @param targetY y value of the target pixel
     */
    private void transPixel(Vec2d.Real regVec, Vec2d.Real sourceVec,
            Vec2d.Real xTransVec, Vec2d.Real yTransVec, int width, int height,
            int targetX, int targetY, int offsetX, int offsetY) {
        float sourceX = xTransVec.get(targetX, targetY) - offsetX;
        float sourceY = yTransVec.get(targetX, targetY) - offsetY;
        if (sourceX >= 0 && sourceY >= 0 && sourceX < width - 1 && sourceY < height - 1) {
            // setting coordinates
            int left = (int) sourceX;
            int top = (int) sourceY;
            int right = (int) sourceX + 1;
            int bottom = (int) sourceY + 1;
            // calculating overlap area
            float xFragment = sourceX - left;
            float yFragment = sourceY - top;
            float tlFactor = (1 - xFragment) * (1 - yFragment);
            float trFactor = xFragment * (1 - yFragment);
            float blFactor = (1 - xFragment) * yFragment;
            float brFactor = xFragment * yFragment;
            // calculating pixel values
            float tlValue = sourceVec.get(left, top) * tlFactor;
            float trValue = sourceVec.get(right, top) * trFactor;
            float blValue = sourceVec.get(left, bottom) * blFactor;
            float brValue = sourceVec.get(right, bottom) * brFactor;
            // setting value of registered pixel
            regVec.set(targetX - offsetX, targetY - offsetY,
                    tlValue + trValue + blValue + brValue);
        }
    }

    /**
     * reads the transformation vector from a buffered reader
     * @param br buffered FileReader
     * @throws IOException 
     */
    private void readInTransVector(BufferedReader br) throws IOException {
        String line;

        boolean xRead = false;
        boolean yRead = false;
        int xRow = 0;
        int yRow = 0;

        while ((line = br.readLine()) != null) {
            if (line.startsWith("X Trans")) {
                xRead = true;
                yRead = false;
                line = br.readLine();
            } else if (line.startsWith("Y Trans")) {
                xRead = false;
                yRead = true;
                line = br.readLine();
            } else if (line.isEmpty()) {
                xRead = false;
                yRead = false;
            }

            if (xRead) {
                writeRowToTransVec(line, reconXTransVec, wfXTransVec, xRow);
                xRow++;
            } else if (yRead) {
                writeRowToTransVec(line, reconYTransVec, wfYTransVec, yRow);
                yRow++;
            }
        }
    }
    
    /**
     * writes a line into transformation vectors
     * @param line line from the transformation file
     * @param reconTransVec transformation vector for reconstructed images
     * @param wfTransVec transformation vector for wide field images
     * @param row Y-Value of the line
     */
    private void writeRowToTransVec(String line, Vec2d.Real reconTransVec, Vec2d.Real wfTransVec, int row) {
        String[] words = line.split(" ");
        int col = 0;
        for (String word : words) {
            try {
                reconTransVec.set(col, row, Float.parseFloat(word));
                if (col % 2 == 0 && row % 2 == 0) {
                    wfTransVec.set(col/2, row/2, Float.parseFloat(word)/2);
                }
                col++;
            } catch (NumberFormatException ex) {
            }
        }
    }
    

}
