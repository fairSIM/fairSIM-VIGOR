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

import org.fairsim.utils.Conf;
import org.fairsim.utils.Tool;
import java.io.File;
import java.io.FileNotFoundException;

/**
 *
 * @author m.lachetta
 */
public class DmdController implements SlmController {
    
    //ControllerServerGui gui;
    
    private DmdController(/*ControllerServerGui serverGui*/) {
        
        //this.gui = serverGui;
        
        // Loading hidapi-Library
        String wd = System.getProperty("user.dir")+"\\";
        String libName = "hidapi";
        System.load(wd+libName+".dll");
        
        // Loading DMD-API-Library
        wd = System.getProperty("user.dir")+"\\";
        libName = "dlp6500-java-api";
        System.load(wd+libName+".dll");
    }
    
    public static String filelist(final Conf.Folder cfg) throws FileNotFoundException {{
        String folder;// = "(not found)";
    
        try {
            folder = Tool.getFile(cfg.getStr("DmdRoFolder").val()).getAbsolutePath();
            System.out.println("Folder"+folder);
        } catch (Conf.EntryNotFoundException ex) {
            folder = System.getProperty("user.dir");
            Tool.error("No folder was found. Folder was set to: " + folder, false);
        }
         File file = new File(folder);
        if (file.exists()) {
            return folder;
        } else {
            throw new FileNotFoundException("Folder does not exists");
        }
        }
    }
    
    String[] filelist = {"C:\\Users\\cwenzel\\Documents\\NetBeansProjects\\fairSIMproject\\vigor-tmp\\linespoints.txt","C:\\Users\\cwenzel\\Documents\\NetBeansProjects\\fairSIMproject\\vigor-tmp\\points.txt"};



    @Override
    public String setRo(int ro) {
        try {
            activateBoard();
            if(isActive()){System.out.println("Board is activ");}
            if(getMode()==2||getMode()==0) setMode(3);
            executeBatchFile(filelist[ro]);
            System.out.println("It was chosen file "+filelist[ro]);
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
            System.out.println("Deactivated current sequence");
            return "Current sequence got deactivated";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        }
    }

    @Override
    public String getRoList() {
        for (String filelist1 : filelist) {
            System.out.println(filelist1);
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
            deactivateBoard();
            System.out.println("Board is inactiv");
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
    
    //connect klappen
    //Modus erst setzen, dann Board ativieren, dann Sequenz starten, letzteres klappt nicht
    //disconnect klappt
    
//    public Boolean isDmd(SlmController slm){
//        Boolean dmd = false;
//        if (slm.connectSlm() =="Connected to the DMD"){
//            dmd=true;
//        }
//        return dmd;
//    }
    
    public static void main(String[] args) throws InterruptedException, DmdException {
        
        DmdController dmd = new DmdController();
        //filelist("C:\\Users\\cwenzel\\Documents\\NetBeansProjects\\fairSIM\\vigor-omx-config-windows.xml");
        
        dmd.getRoList();
        dmd.connectSlm();
        System.out.println("setRo: "+dmd.setRo(0));
        System.out.println("activateRo: "+dmd.activateRo());
        Thread.sleep(2000);
        System.out.println("deactivatRo: "+dmd.deactivateRo());
        Thread.sleep(2000);
        System.out.println("resetBoard: "+dmd.rebootSlm());
        Thread.sleep(5000);
        System.out.println("connecting "+dmd.connectSlm());
        System.out.println("setRo: "+dmd.setRo(1));
        System.out.println("activateRo: "+dmd.activateRo());
        Thread.sleep(2000);
        dmd.disconnectSlm();

    }
    
}
