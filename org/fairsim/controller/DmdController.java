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

import java.io.BufferedReader;
import org.fairsim.utils.Tool;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JOptionPane;

/**
 *
 * @author m.lachetta & c.wenzel
 */
public class DmdController implements SlmController {
    
    ControllerServerGui gui;
    
    public DmdController(ControllerServerGui serverGui) {
        
        this.gui = serverGui;
        
        // Loading hidapi-Library
        String wd = System.getProperty("user.dir")+"\\";
        String libName = "hidapi";
        System.load(wd+libName+".dll");
        
        this.gui.showText("Dmd: loading "+ wd + libName +".dll");
        
        // Loading DMD-API-Library
        wd = System.getProperty("user.dir")+"\\";
        libName = "dlp6500-java-api";
        System.load(wd+libName+".dll");

        this.gui.showText("Dmd: loading "+ wd + libName + ".dll");
    }
    
    public static String[] filelist() throws FileNotFoundException, IOException {
        
        String filename = Tool.getFile(System.getProperty("user.home") + "/NetBeansProjects/fairSIMproject/vigor-tmp").getAbsolutePath();
        File f = new File("C:/Users/cwenzel/Documents/NetBeansProjects/fairSIMproject/vigor-tmp");
        File[] fileArray = f.listFiles();
        String[] stringArray = new String[fileArray.length];
        if(fileArray != null){
            for(int i=0;i<fileArray.length;i++){
                if(fileArray[i].getName().contains("_batch")){
                    if(checkfile(fileArray[i].getAbsoluteFile())){
                        String fileNameWithOutExtension = fileArray[i].getName();
                        int help;
                        help = fileArray[i].getName().lastIndexOf('.');
                        if (help != -1)
                        fileNameWithOutExtension = fileNameWithOutExtension.substring(0, help);
                        stringArray[i] = fileNameWithOutExtension;
                    // filelist.add(fileNameWithOutExtension);        
                    }
                    else{
                        stringArray[i] = "null";
                        System.out.println("file is not compartible 1");
                    }
                }
                else{
                    stringArray[i] = "null";
                    System.out.println("file is not compartible 2");
                }
            }
        }
        else{
            System.out.println("no folder existing");
        }
            final List<String> list =  new ArrayList<String>();
            Collections.addAll(list, stringArray);
            while(list.remove("null"));
            stringArray = list.toArray(new String[list.size()]);
            return stringArray;
        }

    
    public static boolean checkfile(File f) throws FileNotFoundException, IOException{
        boolean boo = false;
        FileReader fr = new FileReader(f.getAbsolutePath());
        BufferedReader br = new BufferedReader(fr);
        int count=1;
        for(int i = 0; i<count;i++){
        String line=br.readLine();
        if(line != null){
           if(line.contains("MBOX_DATA")){
//               System.out.println("MBOX");
               count++;
               boo= true;
           }
           else if(line.contains("PAT_CONFIG")){
//               System.out.println("PAT Config");
               count++;
               boo= true;
           }
           else if(line.contains("PATMEM_LOAD_INIT_MASTER")){
//               System.out.println("PAT LOAD INIT MASTER");
               count++;
               boo= true;
           }
           else if(line.contains("PATMEM_LOAD_DATA_MASTER")){
//               System.out.println("PAT LOAD DATA MASTER");
               count++;
               boo= true;
           }
           else{
               boo= false;
               break;
           }
        }
        }
        
        return boo;
    }
    

    // the board is activated, the mode is tested and perhaps changed, the file is uploaded
    @Override
    public String setRo(int ro) {
        String[] list;
        try {
            activateBoard();
            if(isActive()){System.out.println("Board is activ");}
            if(getMode()==2||getMode()==0) setMode(3);
            list = filelist();
            String file = "C:\\Users\\cwenzel\\Documents\\NetBeansProjects\\fairSIMproject\\vigor-tmp\\"+list[ro]+".txt";
            executeBatchFile(file);
            
           gui.showText("Selected running order '" + ro + "'");
            
            System.out.println("It was chosen file "+file);
            return "File was chosen";
        } catch (DmdException ex) {
            System.out.println("fail in setRo");
            return DmdException.catchedDmdException(ex); 
        }
        catch(IOException ex){
            Logger.getLogger(DmdController.class.getName()).log(Level.SEVERE, null, ex);
            return "building up the file failed";
        }
    }
    
    //the sequence is started and it is tested if its running
    @Override
    public String activateRo() {
        try {
            startSequence();
            if(!isSequenceRunning()){
                return "Sequence coud not be activated.";
                }
            else{
                System.out.println("Activated current sequence in mode "+getMode());
                gui.showText("Activated running order" );
                return "Current sequence got activated"; 
                }
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        }
    }
    
    //the sequence is stopped and it is tested if its running
    @Override
    public String deactivateRo() {
        try {
            stopSequence();
            if(isSequenceRunning()){
                return "Sequence coud not be deactivated.";
            }
            else{
                gui.showText("Deactivated current running order");
                System.out.println("Deactivated current sequence");
            }
            return "Current sequence got deactivated";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        }
    }
    
    //the list of sequences is outlined
    @Override
    public String getRoList() {
        String[] list;
        try {
            list = filelist();
            for(int i = 0; i<list.length;i++){
           gui.showText("ArrayList of sequences constructed");
            System.out.println(list[i]);            
        }
        } catch (IOException ex) {
            Logger.getLogger(DmdController.class.getName()).log(Level.SEVERE, null, ex);//DmdException is not possible here !!
            return "fail to transfer rolist";
        }
        
        return Tool.encodeArray("Transfering rolist", list);
          }
    
    //without this empty String there would be thrown exceptions because of functions from the slm
    @Override
    public String getSlmInfo() {
        String string = " ; ; ; ; ; ";
        return string;
    }

    //it happens a new start of the DMD
    @Override
    public String rebootSlm() {
        try {
            deactivateBoard();
            disconnect();
            Thread.sleep(5000);
            connect();
            gui.showText("Connection to the Dmd opened.");
            System.out.println("DMD is rebooting");
            gui.showText("Dmd is rebooting");
            return "Reboot of the DMD. This may takes more than 2 seconds";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        } catch (InterruptedException ex) {
            Logger.getLogger(DmdController.class.getName()).log(Level.SEVERE, null, ex);
            return "rebooting the Dmd failed";
        }
    }

    //the connection to the DMD is builded
    @Override
    public String connectSlm() {
        try {
            connect();
            System.out.println("Connection to the DMD opened.");
            gui.showText("Connection to the Dmd opened.");
            return "Connected to the DMD";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        }
    }
    
    //the connection to the DMD is lost
    @Override
    public String disconnectSlm() {
        try {
            deactivateBoard();
            System.out.println("Board is inactiv");
            disconnect();
            System.out.println("Disconnection from the DMD.");
            gui.showText("Deactivation of the board was successful.");
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
   
    
    
    public int einlesen() {
 String eingabe = JOptionPane.showInputDialog("Zahl eingeben!", "1");
 try {
 int zahl= Integer.parseInt(eingabe);
 return zahl;
 } catch (NumberFormatException e) {
 System.out.println(eingabe + " ist keine Zahl");
 return(-1);
 }
}
    
    public static void main(String[] args) throws InterruptedException, DmdException, IOException {
        
        String[] list = filelist();
        System.out.println("los gehts");
        for(int i=0;i<list.length;i++){
            System.out.println(list[i]);
        }
        
       
//        DmdController dmd = new DmdController();
//
//        dmd.connectSlm();
//        dmd.getRoList();
//        
//        dmd.setRo(2);
//        dmd.activateRo();
//        Thread.sleep(2000);
//        dmd.deactivateRo();
//        dmd.rebootSlm();
//        Thread.sleep(5000);
//        dmd.setRo(dmd.einlesen());
//        dmd.activateRo();
//        Thread.sleep(2000);
//        dmd.disconnectSlm();

    }
    
}
