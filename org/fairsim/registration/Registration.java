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
import java.util.logging.Level;
import java.util.logging.Logger;
import org.fairsim.linalg.*;
import org.fairsim.utils.Conf;
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class Registration {

    private String channel;
    private final VectorFactory vf;
    private int reconWidth;
    private int reconHeight;
    private int wfWidth;
    private int wfHeight;
    private Vec2d.Real reconXTransVec;
    private Vec2d.Real reconYTransVec;
    private Vec2d.Real wfXTransVec;
    private Vec2d.Real wfYTransVec;
    static private int threads;
    static public final List<Registration> REGISTRATIONS;
    static private boolean widefield;
    static private boolean recon;

    static {
        REGISTRATIONS = new ArrayList<Registration>();
    }
    
    /**
     * Constructor for a registration object
     * @param file raw registration file of bUnwrappedJ
     * @throws IOException is throwen if the inputfile does not exist or not
     * have the cerrect structure/format
     */
    Registration(String file) throws IOException {
        vf = Vec.getBasicVectorFactory();
        reconWidth = -1;
        reconHeight = -1;
        wfWidth = -1;
        wfHeight = -1;
        reconXTransVec = null;
        reconYTransVec = null;
        wfXTransVec = null;
        wfYTransVec = null;

        BufferedReader br = new BufferedReader(new FileReader(file));
        System.out.println("[fairSIM] Registration: Readin registration file: " + file);

        reconWidth = Integer.parseInt(br.readLine().split("=")[1]);
        reconHeight = Integer.parseInt(br.readLine().split("=")[1]);
        wfWidth = reconWidth / 2;
        wfHeight = reconHeight / 2;

        reconXTransVec = vf.createReal2D(reconWidth, reconHeight);
        reconYTransVec = vf.createReal2D(reconWidth, reconHeight);
        wfXTransVec = vf.createReal2D(wfWidth, wfHeight);
        wfYTransVec = vf.createReal2D(wfWidth, wfHeight);
        readInTransVector(br);

        br.close();
        System.out.println("[fairSIM] Registration: Readin done for: " + file);

        threads = Runtime.getRuntime().availableProcessors();
        if (threads > wfHeight) {
            threads = wfHeight;
        }
        if (threads < 1) {
            threads = 1;
        }
    }

    /**
     * Creates a new Registration and adds it to the list of registrations
     * @param cfg configuration to get the folder for the .txt files with the
     * raw registrations
     * @param channel channel (wavelenght) of the fairsim-software and registration
     */
    public static void createRegistration(final Conf.Folder cfg, final String channel) {
        new Thread(new Runnable() {
            @Override
            public void run() {
                String regFolder = "(not found)";
                try {
		    regFolder = cfg.getStr("RegistrationFolder").val();
		    String filename = Tool.getFile(regFolder + channel + ".txt").getAbsolutePath();
                    Registration reg = new Registration( filename );
                    reg.channel = channel;
                    REGISTRATIONS.add(reg);
                } catch (Conf.EntryNotFoundException ex) {
                    System.out.println("[fairSIM] Registration: No registration folder found");
                } catch (IOException ex) {
                    System.out.println("[fairSIM] Registration: No registration possible for channel: " + channel );
		    System.out.println("[fairSIM] "+ex);
		}
            }

        }).start();
    }
    
    /**
     * Finds the required registration of the specific channel
     * @param channel channel of the required registration
     * @return registration objekt for a specific channel
     * @throws NoSuchFieldException is thrown if the required registration
     * does not exists
     */
    public static Registration getRegistration(String channel) throws NoSuchFieldException {
        for (Registration reg : REGISTRATIONS) {
            if (reg.channel.equals(channel)) {
                return reg;
            }
        }
        throw new NoSuchFieldException();
    }
    
    /**
     * (de)activates the registration in widefield
     * @param b 
     */
    static void setWidefield(boolean b) {
        widefield = b;
    }
    
    /**
     * (de)activates the registration in reconstruction
     * @param b 
     */
    static void setRecon(boolean b) {
        recon = b;
    }
    
    /**
     * 
     * @return is registration in widefield active?
     */
    public static boolean isWidefield() {
        return widefield;
    }
    
    /**
     * 
     * @return is registration in widefield ative?
     */
    public static boolean isRecon() {
        return recon;
    }
    
    
    public Vec2d.Real registerImageOld(final Vec2d.Real sourceVec, final char type) {
       final int width;
       final int height;
       
        switch (type) {
            case 'r':
                width = reconWidth;
                height = reconHeight;
                break;
            case 'w':
                width = wfWidth;
                height = wfHeight;
                break;
            default:
                throw new IllegalArgumentException("Only 'r' and 'w' allowed as type");
        }
       
        
        if (sourceVec.vectorWidth() != width || sourceVec.vectorHeight() != height) {
            Tool.trace("Image registration '" + type + "' failed, because of differences in diminsions");
            if (type == 'r') {
                setRecon(false);
            } else if (type == 'w') {
                setWidefield(false);
            }
        }
        final Vec2d.Real regVec = vf.createReal2D(width, height); 
        
        //Multi-Threaded way
        final int blockSize = height / threads;
        Thread[] blocks = new Thread[threads];

        for (int threadId = 0; threadId < threads; threadId++) {
            final int tId = threadId;
            blocks[tId] = new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int y = blockSize * tId; y < blockSize * (tId + 1); y++) {
                        for (int x = 0; x < width; x++) {
                            transPixelOld(regVec, sourceVec, x, y, type);
                        }
                    }
                }
            });
            blocks[tId].start();
        }

        
        for (int y = blockSize * threads; y < height; y++) {
            for (int x = 0; x < width; x++) {
                transPixelOld(regVec, sourceVec, x, y, type);
            }
        } 
        
        for (int threadId = 0; threadId < threads; threadId++) {
            try {
                blocks[threadId].join();
            } catch (InterruptedException ex) {
                System.err.println("Registration thread interrupted!");
                Logger.getLogger(Registration.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        return regVec;
    }
    
    private void transPixelOld(Vec2d.Real regVec, Vec2d.Real sourceVec, int sourceX, int sourceY, char type) {
        float sourceValue = sourceVec.get(sourceX, sourceY);
        float targetX;
        float targetY;
        int width;
        int height;
        
        switch (type) {
            case 'r':
                width = reconWidth;
                height = reconHeight;
                targetX = reconXTransVec.get(sourceX, sourceY);
                targetY = reconYTransVec.get(sourceX, sourceY);
                break;
            case 'w':
                width = wfWidth;
                height = wfHeight;
                targetX = wfXTransVec.get(sourceX, sourceY);
                targetY = wfYTransVec.get(sourceX, sourceY);
                break;
            default:
                throw new IllegalArgumentException("Only 'r' and 'w' allowed as type");
        }

        if (targetX >= 0 && targetY >= 0 && targetX < width - 1 && targetY < height - 1) {
            int left = (int) targetX;
            int top = (int) targetY;
            int right = (int) targetX + 1;
            int bottom = (int) targetY + 1;

            float xFragment = targetX - left;
            float yFragment = targetY - top;

            float tlFactor = (1 - xFragment) * (1 - yFragment);
            float trFactor = xFragment * (1 - yFragment);
            float blFactor = (1 - xFragment) * yFragment;
            float brFactor = xFragment * yFragment;
            
            float tlValue = regVec.get(left, top);
            float trValue = regVec.get(right, top);
            float blValue = regVec.get(left, bottom);
            float brValue = regVec.get(right, bottom);
            
            tlValue += sourceValue * tlFactor;
            trValue += sourceValue * trFactor;
            blValue += sourceValue * blFactor;
            brValue += sourceValue * brFactor;
            
            regVec.set(left, top, tlValue);
            regVec.set(right, top, trValue);
            regVec.set(left, bottom, blValue);
            regVec.set(right, bottom, brValue);
        }
    }
    
    public Vec2d.Real registerWfImageNew(Vec2d.Real sourceVec) {
        if (sourceVec.vectorWidth() != wfWidth || sourceVec.vectorHeight() != wfHeight) {
            Tool.trace("Image registration failed, because of differences in diminsions");
            setWidefield(false);
            return sourceVec;
        }
        return registerImageNew(sourceVec, wfXTransVec, wfYTransVec, wfWidth, wfHeight);
    }
    
    public Vec2d.Real registerReconImageNew(Vec2d.Real sourceVec) {
        if (sourceVec.vectorWidth() != reconWidth || sourceVec.vectorHeight() != reconHeight) {
            Tool.trace("Image registration failed, because of differences in diminsions");
            setRecon(false);
            return sourceVec;
        }
        return registerImageNew(sourceVec, reconXTransVec, reconYTransVec, reconWidth, reconHeight);
    }
    
    private Vec2d.Real registerImageNew(final Vec2d.Real sourceVec, 
	final Vec2d.Real xTransVec, final Vec2d.Real yTransVec, 
	final int width, final int height) {
        
        final Vec2d.Real regVec = vf.createReal2D(width, height); 
        
        //Multi-Threaded way
        final int blockSize = height / threads;
        final Thread[] blocks = new Thread[threads];

        for (int threadId = 0; threadId < threads; threadId++) {
            final int tId = threadId;
            blocks[tId] = new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int y = blockSize * tId; y < blockSize * (tId + 1); y++) {
                        for (int x = 0; x < width; x++) {
                            transPixelNew(regVec, sourceVec, xTransVec, yTransVec, width, height, x, y);
                        }
                    }
                }
            });
            blocks[tId].start();
        }

        for (int y = blockSize * threads; y < height; y++) {
            for (int x = 0; x < width; x++) {
                transPixelNew(regVec, sourceVec, xTransVec, yTransVec, width, height, x, y);
            }
        } 
        
        for (int threadId = 0; threadId < threads; threadId++) {
            try {
                blocks[threadId].join();
            } catch (InterruptedException ex) {
                System.err.println("Registration thread interrupted!");
                Logger.getLogger(Registration.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        return regVec;
    }
    
    private void transPixelNew(Vec2d.Real regVec, Vec2d.Real sourceVec, Vec2d.Real xTransVec, Vec2d.Real yTransVec, int width, int height, int sourceX, int sourceY) {
        float sourceValue = sourceVec.get(sourceX, sourceY);
        float targetX = xTransVec.get(sourceX, sourceY);
        float targetY = yTransVec.get(sourceX, sourceY);;

        if (targetX >= 0 && targetY >= 0 && targetX < width - 1 && targetY < height - 1) {
            int left = (int) targetX;
            int top = (int) targetY;
            int right = (int) targetX + 1;
            int bottom = (int) targetY + 1;

            float xFragment = targetX - left;
            float yFragment = targetY - top;

            float tlFactor = (1 - xFragment) * (1 - yFragment);
            float trFactor = xFragment * (1 - yFragment);
            float blFactor = (1 - xFragment) * yFragment;
            float brFactor = xFragment * yFragment;
            
            float tlValue = regVec.get(left, top);
            float trValue = regVec.get(right, top);
            float blValue = regVec.get(left, bottom);
            float brValue = regVec.get(right, bottom);
            
            tlValue += sourceValue * tlFactor;
            trValue += sourceValue * trFactor;
            blValue += sourceValue * blFactor;
            brValue += sourceValue * brFactor;
            
            regVec.set(left, top, tlValue);
            regVec.set(right, top, trValue);
            regVec.set(left, bottom, blValue);
            regVec.set(right, bottom, brValue);
        }
    }
    
    /**
     * Registers a widefield image
     * @param sourceVec input vektor for the registration
     * @return output vektor from the registration
     */
    public Vec2d.Real registerWfImageInverse(Vec2d.Real sourceVec) {
        if (sourceVec.vectorWidth() != wfWidth || sourceVec.vectorHeight() != wfHeight) {
            Tool.trace("Image registration failed, because of differences in diminsions");
            setWidefield(false);
            return sourceVec;
        }
        return registerImageInverse(sourceVec, wfXTransVec, wfYTransVec, wfWidth, wfHeight);
    }
    
    /**
     * Registers a reconstructed Vektor
     * @param sourceVec input vektor for the registration
     * @return output vektor from the registration
     */
    public Vec2d.Real registerReconImageInverse(Vec2d.Real sourceVec) {
        if (sourceVec.vectorWidth() != reconWidth || sourceVec.vectorHeight() != reconHeight) {
            Tool.trace("Image registration failed, because of differences in diminsions");
            setRecon(false);
            return sourceVec;
        }
        return registerImageInverse(sourceVec, reconXTransVec, reconYTransVec, reconWidth, reconHeight);
    }
    
    /**
     * Registers a Vektor
     * @param sourceVec input Vektor for the registration
     * @param xTransVec transformation Vector in X
     * @param yTransVec transformatio Vector in Y
     * @param width width of the Vector
     * @param height height of the Vector
     * @return output vektor from the registration
     */
    Vec2d.Real registerImageInverse(final Vec2d.Real sourceVec, 
	final Vec2d.Real xTransVec, final Vec2d.Real yTransVec, 
	final int width, final int height) {
        
        final Vec2d.Real regVec = vf.createReal2D(width, height); 
        
        //Multi-Threaded way
        final int blockSize = height / threads;
        final Thread[] blocks = new Thread[threads];

        for (int threadId = 0; threadId < threads; threadId++) {
            final int tId = threadId;
            blocks[tId] = new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int y = blockSize * tId; y < blockSize * (tId + 1); y++) {
                        for (int x = 0; x < width; x++) {
                            transPixelInverse(regVec, sourceVec, xTransVec, yTransVec, width, height, x, y);
                        }
                    }
                }
            });
            blocks[tId].start();
        }

        for (int y = blockSize * threads; y < height; y++) {
            for (int x = 0; x < width; x++) {
                transPixelInverse(regVec, sourceVec, xTransVec, yTransVec, width, height, x, y);
            }
        } 
        
        for (int threadId = 0; threadId < threads; threadId++) {
            try {
                blocks[threadId].join();
            } catch (InterruptedException ex) {
                System.err.println("Registration thread interrupted!");
                Logger.getLogger(Registration.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        
        return regVec;
    }
    
    /**
     * Registers a single pixel in the output vektor from the registration
     * @param regVec output vector from the registration
     * @param sourceVec input Vektor for the registration
     * @param xTransVec transformation Vector in X
     * @param yTransVec transformatio Vector in Y
     * @param width width of the Vector
     * @param height height of the Vector
     * @param targetX x value of the target pixel
     * @param targetY y value of the target pixel
     */
    private void transPixelInverse(Vec2d.Real regVec, Vec2d.Real sourceVec, Vec2d.Real xTransVec, Vec2d.Real yTransVec, int width, int height, int targetX, int targetY) {
        float sourceX = xTransVec.get(targetX, targetY);
        float sourceY = yTransVec.get(targetX, targetY);

        if (sourceX >= 0 && sourceY >= 0 && sourceX < width - 1 && sourceY < height - 1) {
            int left = (int) sourceX;
            int top = (int) sourceY;
            int right = (int) sourceX + 1;
            int bottom = (int) sourceY + 1;
            
            float xFragment = sourceX - left;
            float yFragment = sourceY - top;

            float tlFactor = (1 - xFragment) * (1 - yFragment);
            float trFactor = xFragment * (1 - yFragment);
            float blFactor = (1 - xFragment) * yFragment;
            float brFactor = xFragment * yFragment;

            float tlValue = sourceVec.get(left, top) * tlFactor;
            float trValue = sourceVec.get(right, top) * trFactor;
            float blValue = sourceVec.get(left, bottom) * blFactor;
            float brValue = sourceVec.get(right, bottom) * brFactor;
            
            regVec.set(targetX, targetY, tlValue + trValue + blValue + brValue);
        }
    }

    /**
     * reads the transformation vector from a buffered reader
     * @param br buffered filereader
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
     * @param reconTransVec transformation vector for recunstructed images
     * @param wfTransVec transformation vector for widefield images
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
