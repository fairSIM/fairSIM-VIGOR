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

/**
 *
 * @author m.lachetta & c.wenzel
 */
public class DmdController implements SlmController {

    AbstractServer.ServerGui gui;
    int selectedRo;
    String[] ros;
    boolean busy;

    public DmdController(AbstractServer.ServerGui serverGui) {

        this.gui = serverGui;
        this.selectedRo = -1;
        this.ros = new String[1];
        this.busy = false;
        // Loading hidapi-Library
        String wd = System.getProperty("user.dir") + "\\";
        String libName = "hidapi";
        System.load(wd + libName + ".dll");

        this.gui.showText("Dmd: loading " + wd + libName + ".dll");

        // Loading DMD-API-Library
        wd = System.getProperty("user.dir") + "\\";
        libName = "dlp6500-java-api";
        System.load(wd + libName + ".dll");

        this.gui.showText("Dmd: loading " + wd + libName + ".dll");
    }

    /**
     * A list of checked files is builded 
     * @return Stringarray of filenames
     * @throws FileNotFoundException
     * @throws IOException 
     */
    public static String[] filelist() throws FileNotFoundException, IOException {

        File f = new File("C:/Users/cwenzel/Documents/NetBeansProjects/fairSIMproject/vigor-tmp");
        File[] fileArray = f.listFiles();
        String[] stringArray = new String[fileArray.length];
        if (fileArray != null) {
            for (int i = 0; i < fileArray.length; i++) {
                if (fileArray[i].getName().contains("_batch")) {
                    if (checkfile(fileArray[i].getAbsoluteFile())) {
                        String fileNameWithOutExtension = fileArray[i].getName();
                        int help;
                        help = fileArray[i].getName().lastIndexOf('.');
                        if (help != -1) {
                            fileNameWithOutExtension = fileNameWithOutExtension.substring(0, help);
                        }
                        stringArray[i] = fileNameWithOutExtension;   
                    } else {
                        stringArray[i] = "null";
                    }
                } else {
                    stringArray[i] = "null";
                }
            }
        } else {
            System.err.println("no folder existing");
        }
        final List<String> list = new ArrayList<String>();
        Collections.addAll(list, stringArray);
        while (list.remove("null"));
        stringArray = list.toArray(new String[list.size()]);
        return stringArray;
    }


    /**
     * this function checks the files, which came from filelist()
     * @param f
     * @return boolean which checks if the file fit into the pattern and if it can be opened
     * @throws FileNotFoundException if file does not fit into the pattern
     * @throws IOException if file could not be open
     */
    public static boolean checkfile(File f) throws FileNotFoundException, IOException {
        boolean boo = false;
        FileReader fr = new FileReader(f.getAbsolutePath());
        BufferedReader br = new BufferedReader(fr);
        int count = 1;
        for (int i = 0; i < count; i++) {
            String line = br.readLine();
            if (line != null) {
                if (line.contains("MBOX_DATA")) {
                    count++;
                    boo = true;
                } else if (line.contains("PAT_CONFIG")) {
                    count++;
                    boo = true;
                } else if (line.contains("PATMEM_LOAD_INIT_MASTER")) {
                    count++;
                    boo = true;
                } else if (line.contains("PATMEM_LOAD_DATA_MASTER")) {
                    count++;
                    boo = true;
                } else {
                    boo = false;
                    break;
                }
            }
        }
        return boo;
    }
    
    /**
     * the board is activated, the mode is tested and perhaps changed, the file is uploaded
     * @param ro which defines the running number which is chosen
     * @return String 
     * @throws DmdException
     */
    @Override
    public String setRo(int ro) {
        if (busy) {
            return "Error: Dmd is busy";
        }
        String[] list;
        this.selectedRo = ro;
        try {
            if (!isActive()) {
                activateBoard();
            }

            if (getMode() == 2 || getMode() == 0) {
                setMode(3);
            }

            list = ros;

            String file = "C:\\Users\\cwenzel\\Documents\\NetBeansProjects\\fairSIMproject\\vigor-tmp\\" + list[ro] + ".txt";
            
            if(isSequenceRunning()){return "Error: stop sequence before starting a new one";}
            
            new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        busy = true;
                        executeBatchFile(file);
                        busy = false;
                    } catch (DmdException ex) {
                        System.err.println("Exception in setRo, executing batch file, why?");
                        ex.printStackTrace();
                    }

                }
            }).start();

            gui.showText("Selected running order '" + ro + "'");

            return "File was chosen";
        } catch (DmdException ex) {
            System.err.println("Exception in the setRo-function from the Dmd, often a problem with (de-)aktivation of the board");
            return DmdException.catchedDmdException(ex);
        }
    }


    /**
     * the sequence is started and it is tested if its running
     * @return String
     * @ throws DmdException
     */
    @Override
    public String activateRo() {
        if (busy) {
            return "Error: Dmd is busy";
        }
        try {
            //Error if no sequence is selected
            if (this.selectedRo == -1) {
                return "Error: No ro selected";
            }
            if (isActive()) {
                startSequence();
                if (!isSequenceRunning()) {
                    return "Sequence coud not be activated.";
                } else {
                    gui.showText("Activated running order");
                    return "Current sequence got activated";
                }
            } else {
                return "Error: Board was not activated, because there was no sequence set.";
            }
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        }
    }
    
    /**
     * the sequence is stopped and it is tested if its running
     * @return String
     * @throws DmdException
     */
    @Override
    public String deactivateRo() {
        if (busy) {
            return "Error: Dmd is busy";
        }
        try {
            if (isActive()) {
                stopSequence();
            }
            if (isSequenceRunning()) {
                return "Sequence coud not be deactivated.";
            } else {
                gui.showText("Deactivated current running order");         
            }
            return "Current sequence got deactivated";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        }
    }

    /**
     * the list of sequences is outlined
     * @return String
     */
    @Override
    public String getRoList() {
        String[] list;
        list = ros;
        if (ros[0] != null) {
            gui.showText("List of sequences constructed");
            return Tool.encodeArray("Transfering rolist", list);
        } else {
            return "Error: List of sequences is empty";
        }
    }
    
    /**
     * 
     * @return selected Ro
     */
    @Override
    public String getSlmSelectedRo() {
        int select = this.selectedRo;
        return "Transfering info;" + select;
    }
    
    /**
     * Dmd is rebootet
     * @return String
     * @throws InterruptedException if Thread is interrupted
     * @throws DmdException if problem with (dis-)connection
     */
    @Override
    public String rebootSlm() {
        if (busy) {
            return "Error: Dmd is busy";
        }
        new Thread(new Runnable() {
            public void run() {
                try {
                    busy = true;
                    rebootSlm();
                    Thread.sleep(2000);
                    disconnect();
                    connect();
                    busy = false;
                } catch (DmdException ex) {
                    System.err.println("Exception in rebootSlm, disconnect/connect, why?");
                    ex.printStackTrace();
                } catch (InterruptedException ex) {
                    System.err.println("Exception in rebootSlm, Thread.sleep, why?");
                    ex.printStackTrace();
                }
            }
        }).start();
        gui.showText("Dmd is rebooting");
        return "Reboot of the DMD. This may takes around 20 seconds";
    }

    /**
     * the connection to the DMD is builded
     * @return String
     * @throws DmdException if connection fails
     * @throws IOException if filelist() fails
     */
    @Override
    public String connectSlm() {
        if (busy) {
            return "Error: Dmd is busy";
        }
        try {
            connect();
            selectedRo = -1;
            ros = filelist();
            gui.showText("Connection to the Dmd opened.");
            return "Connected to the DMD";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);
        } catch (IOException ex) {
            return "Error: Can not read files.";
        }
    }
    
    /**
     * the connection to the DMD is lost
     * @return String
     * @throws DmdException
     */
    @Override
    public String disconnectSlm() {
        if (busy) {
            return "Error: Dmd is busy";
        }
        try {
            if(isActive()){
              deactivateBoard();  
            }
            disconnect();
            gui.showText("Deactivation of the board was successful.");
            return "Disconnected from the DMD";
        } catch (DmdException ex) {
            return DmdException.catchedDmdException(ex);

        }
    }

    /**
     * Exceptions thrown from the dlp6500-java-api.dll
     */
    static class DmdException extends Exception {

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
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private boolean isSequenceRunning() throws DmdException;

    /**
     * opens the connection to the dmd
     *
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private void connect() throws DmdException;

    /**
     * closes the connection to the dmd
     *
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private void disconnect() throws DmdException;

    /**
     * activate the controller board of the dmd
     *
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private void activateBoard() throws DmdException;

    /**
     * resets/reboots the controller board of the dmd, requires ~2 seconds
     *
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private void resetBoard() throws DmdException;

    /**
     * deactivates the controller board
     *
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private void deactivateBoard() throws DmdException;

    /**
     *
     * @return true if controller board is active, else false
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private boolean isActive() throws DmdException;

    /**
     * sets the operating mode of the controller board
     *
     * @param mode mode id 0 - Video Mode <br>
     * 1 - Pre-stored Pattern Mode <br>
     * 2 - Video Pattern Mode, throws exceptio w/o video source <br>
     * 3 - Pattern On-The-Fly Mode, recommended
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private void setMode(int mode) throws DmdException;

    /**
     *
     * @return the operating mode of the controller board 0 - Video Mode <br>
     * 1 - Pre-stored Pattern Mode <br>
     * 2 - Video Pattern Mode <br>
     * 3 - Pattern On-The-Fly Mode
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private int getMode() throws DmdException;

    /**
     * loads and executes a batch file
     *
     * @param file
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private void executeBatchFile(String file) throws DmdException;

    /**
     * starts the loaded sequence on the dmd
     *
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private void startSequence() throws DmdException;

    /**
     * pause the sequence on the dmd
     *
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private void pauseSequence() throws DmdException;

    /**
     * stops the sequence on the dmd
     *
     * @throws org.fairsim.controller.DmdController.DmdException if something
     * went wrong
     */
    native private void stopSequence() throws DmdException;

}
