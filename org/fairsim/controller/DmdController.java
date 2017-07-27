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
public class DmdController implements SlmController {
    
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
    
    String[] filelist = {"C:\\Users\\cwenzel\\Desktop\\Texas Instruments Software\\Texas Instruments Software\\DLPC900REF-SW-3.0.0\\DLPC900REF-GUI\\LCR6500_Images\\linespoints.txt","C:\\Users\\cwenzel\\Desktop\\Texas Instruments Software\\Texas Instruments Software\\DLPC900REF-SW-3.0.0\\DLPC900REF-GUI\\LCR6500_Images\\points.txt"};

    @Override
    public String setRo(int ro) {
        try {
            if(getMode()==2||getMode()==0) setMode(3);
            executeBatchFile(filelist[ro]);
            System.out.println("It was chosen file "+filelist[ro]);
            activateBoard();
            if(isActive()){System.out.println("Board is activ");}
            return "File was chosen";
        } catch (DmdException ex) {
            System.out.println("fail in setRo");
            return DmdException.catchedDmdException(ex); 
        }
    }

    @Override
    public String activateRo() {
        try {
            startSequence();
            if(!isSequenceRunning()){
                return "Sequence coud not be activated.";
                }
            else{
                System.out.println("Activated current sequence in mode "+getMode());
                return "Current sequence got activated"; 
                }
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        }
    }

    @Override
    public String deactivateRo() {
        try {
            stopSequence();
            deactivateBoard();
            System.out.println("Deactivated current sequence");
            return "Current sequence got deactivated";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        }
    }

    @Override
    public String getRoList() {
            for(int i=0;i<filelist.length;i++){
            System.out.println(filelist[i]); 
            }
            return "Filelist has been output completely.";
          }

    @Override
    public String getSlmInfo() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String rebootSlm() {
        try {
            resetBoard();
            System.out.println("DMD is rebooting");
            return "Reboot of the DMD. This may takes more than 2 seconds";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        }
    }

    @Override
    public String connectSlm() {
        try {
            connect();
            System.out.println("Connection to the DMD opened.");
                return "Connected to the DMD";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        }
    }

    @Override
    public String disconnectSlm() {
        try {
            disconnect();
            System.out.println("Disconnection from the DMD.");
                return "Disconnected from the DMD";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
            
        }    
    }
    
    /**
     * Exceptions thrown from the dlp6500-java-api.dll
     */
    static class DmdException extends Exception{
        
        DmdException(String message) {
            super(message);
        }
        static String catchedDmdException(DmdException ex) {
             return "Error: " + ex.getMessage() + "  ;  " + ex.getClass();
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
    
    //connect sollte klappen
    //Modus erst setzen, dann Board ativieren, dann Sequenz starten, letzteres klappt nicht
    //disconnect + deactivate klappt dann setzt auch Geräusch aus
    
    public static void main(String[] args) throws InterruptedException, DmdException {
        DmdController dmd = new DmdController();
        
        dmd.connectSlm();
        dmd.deactivateBoard();
       
//        dmd.isActive();
//        dmd.activateBoard();
        
//        dmd.setMode(1);
//        dmd.executeBatchFile("C:\\Users\\cwenzel\\Documents\\NetBeansProjects\\fairSIMproject\\vigor-tmp\\points.txt");
      //  dmd.setRo(1);
//        dmd.startSequence();
//        dmd.activateBoard();
//        dmd.isActive();
//        System.out.println("isActiv"+dmd.isActive());
        
      
//        System.out.println("isSequenceRunning: " + dmd.isSequenceRunning());
//        dmd.deactivateRo();
//        dmd.activateRo();
//        System.out.println("isActive: " + dmd.isActive());
//        
//        
//        dmd.setRo(1);
//        System.out.println("getMode: " + dmd.getMode());
//        dmd.setMode(3);
//        System.out.println("getMode: " + dmd.getMode());
//        //dmd.setRo(0);
//        //dmd.executeBatchFile("C:\\Users\\cwenzel\\Desktop\\Texas Instruments Software\\Texas Instruments Software\\DLPC900REF-SW-3.0.0\\DLPC900REF-GUI\\LCR6500_Images\\testbatch.txt");
//        
//        dmd.resetBoard();
//        Thread.sleep(2000);
//        dmd.connectSlm();
//        System.out.println("isSequenceRunning: " + dmd.isSequenceRunning());
//        dmd.deactivateRo();
//        dmd.activateRo();
//        System.out.println("isActive: " + dmd.isActive());
//        
//        
//        dmd.setMode(1);
//        System.out.println("getMode: " + dmd.getMode());
//        dmd.setMode(3);
//        System.out.println("getMode: " + dmd.getMode());
//       // dmd.executeBatchFile("C:\\Users\\cwenzel\\Desktop\\Texas Instruments Software\\Texas Instruments Software\\DLPC900REF-SW-3.0.0\\DLPC900REF-GUI\\LCR6500_Images\\testbatch.txt");
//        dmd.startSequence();
//        Thread.sleep(5000);
//        dmd.pauseSequence();
//        Thread.sleep(5000);
//        dmd.startSequence();
//        Thread.sleep(5000);
//        dmd.stopSequence();
//        dmd.deactivateRo();
//        dmd.disconnectSlm();
    }
    
}
