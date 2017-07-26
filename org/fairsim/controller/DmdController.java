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
package org.fairsim.controller;

/**
 *
 * @author m.lachetta
 */
public class DmdController {
    
    ControllerServerGui gui;
    
    private DmdController() {
        // Loading hidapi-Library
        String wd = System.getProperty("user.dir")+"\\";
        String libName = "hidapi";
        System.load(wd+libName+".dll");
        
        // Loading DMD-API-Library
        wd = System.getProperty("user.dir")+"\\";
        libName = "dlp6500-java-api";
        System.load(wd+libName+".dll");
    }
    
    /**
     * Exceptions thrown from the dlp6500-java-api.dll
     */
    static class DmdException extends Exception{
        
        DmdException(String message) {
            super(message);
        }
        
    }
    
    /**
     * 
     * @return true if a sequence is running, else false
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private boolean isSequenceRunning() throws DmdException;
    
    /**
     * opens the connection to the dmd
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private void connect() throws DmdException;
    
    /**
     * closes the connection to the dmd
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private void disconnect() throws DmdException;
    
    /**
     * activate the controller board of the dmd
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private void activateBoard() throws DmdException;
    
    /**
     * resets/reboots the controller board of the dmd, requires ~2 seconds
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private void resetBoard() throws DmdException;
    
    /**
     * deactivates the controller board
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private void deactivateBoard() throws DmdException;
    
    /**
     * 
     * @return true if controller board is active, else false
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private boolean isActive() throws DmdException;
    
    /**
     * sets the operating mode of the controller board
     * @param mode mode id
     * 0 - Video Mode <br>
     * 1 - Pre-stored Pattern Mode <br>
     * 2 - Video Pattern Mode, throws exceptio w/o video source <br>
     * 3 - Pattern On-The-Fly Mode, recommended
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private void setMode(int mode) throws DmdException;
    
    /**
     * 
     * @return the operating mode of the controller board
     * 0 - Video Mode <br>
     * 1 - Pre-stored Pattern Mode <br>
     * 2 - Video Pattern Mode <br>
     * 3 - Pattern On-The-Fly Mode
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private int getMode() throws DmdException;
    
    /**
     * loads and executes a batch file
     * @param file
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private void executeBatchFile(String file) throws DmdException;
    
    /**
     * starts the loaded sequence on the dmd
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private void startSequence() throws DmdException;
    
    /**
     * pause the sequence on the dmd
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private void pauseSequence() throws DmdException;
    
    /**
     * stops the sequence on the dmd
     * @throws org.fairsim.controller.DmdController.DmdException if something went wrong
     */
    native private void stopSequence() throws DmdException;
    
    public static void main(String[] args) throws InterruptedException, DmdException {
        DmdController dmd = new DmdController();
        dmd.connect();
        System.out.println("isSequenceRunning: " + dmd.isSequenceRunning());
        dmd.deactivateBoard();
        dmd.activateBoard();
        System.out.println("isActive: " + dmd.isActive());
        
        
        dmd.setMode(0);
        System.out.println("getMode: " + dmd.getMode());
        dmd.setMode(3);
        System.out.println("getMode: " + dmd.getMode());
        dmd.executeBatchFile("D:\\SLMs\\Texas Instruments Software\\DLPC900REF-SW-3.0.0\\DLPC900REF-GUI\\LCR6500_Images\\loadTextSeq_bat.txt");
        
        dmd.resetBoard();
        Thread.sleep(2000);
        dmd.connect();
        System.out.println("isSequenceRunning: " + dmd.isSequenceRunning());
        dmd.deactivateBoard();
        dmd.activateBoard();
        System.out.println("isActive: " + dmd.isActive());
        
        
        dmd.setMode(0);
        System.out.println("getMode: " + dmd.getMode());
        dmd.setMode(3);
        System.out.println("getMode: " + dmd.getMode());
        dmd.executeBatchFile("D:\\SLMs\\Texas Instruments Software\\DLPC900REF-SW-3.0.0\\DLPC900REF-GUI\\LCR6500_Images\\loadTextSeq_bat.txt");
        dmd.startSequence();
        Thread.sleep(5000);
        dmd.pauseSequence();
        Thread.sleep(5000);
        dmd.startSequence();
        Thread.sleep(5000);
        dmd.stopSequence();
        dmd.disconnect();
    }
    
}
