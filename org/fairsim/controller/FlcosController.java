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

import com.forthdd.commlib.exceptions.AbstractException;
import com.forthdd.commlib.exceptions.CommException;
import com.forthdd.commlib.r4.R4CommLib;
import com.forthdd.commlib.core.CommLib;
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class FlcosController implements SlmController {
    
    ControllerServerGui gui;
   
    /**
     * Constructor for the SLM-Controller
     * @param serverGui 
     */
    FlcosController(ControllerServerGui serverGui) {
        this.gui = serverGui;
        
        // Loding FLCoS-API-Library
        String wd = System.getProperty("user.dir")+"/";
        String libName = "R4CommLib";
        this.gui.showText("Flcos: loading "+libName+".dll from "+wd);
        System.load(wd+libName+".dll");
    }
    
    
    /**
     * passes not catched exceptions to the client
     *
     * @param ex thrown Exception
     * @param out String-output-Stream of the server
     */
    private String catchedAbstractException(AbstractException ex) {
        return "Error: " + ex.getMessage() + "  ;  Code: " + ex.getCode() + "  ;  " + ex.getClass();
    }

    /**
     * sets a new selected running order
     *
     * @param ro new running order
     * @param out String-output-Stream of the server
     */
    @Override
    public String setRo(int ro) {
        //out.println("Try to set running order to: " + ro);
        try {
            R4CommLib.rpcRoDeactivate();
            R4CommLib.rpcRoSetSelected(ro);
            gui.showText("Selected running order '" + ro + "'");
            return "Running order was set to: " + ro;
        } catch (AbstractException ex) {
            return catchedAbstractException(ex);
        }
    }

    /**
     * activates the selected running order
     *
     * @param out String-output-Stream of the server
     */
    @Override
    public String activateRo() {
        //out.println("Try to activate selected running order");
        try {
            R4CommLib.rpcRoActivate();
            gui.showText("Activated running order '" + R4CommLib.rpcRoGetSelected() + "'");
            return "Selected running order [" + R4CommLib.rpcRoGetSelected() + "] got activated";
        } catch (AbstractException ex) {
            return catchedAbstractException(ex);
        }
    }

    /**
     * deactivates the current running order
     *
     * @param out String-output-Stream of the server
     */
    @Override
    public String deactivateRo() {
        //out.println("Try to deactivate current running order");
        try {
            R4CommLib.rpcRoDeactivate();
            gui.showText("Deactivated current running order");
            return "Current running order got deactivated";
        } catch (AbstractException ex) {
            return catchedAbstractException(ex);
        }
    }

    /**
     * Sends an array of information about the SLM to the client
     *
     * @param out String-output-Stream of the server
     */
    @Override
    public String getSlmSelectedRo() {
        //out.println("Try to transfer Slm Information");
//        try {
//            String[] info = new String[6];
//            info[0] = R4CommLib.libGetVersion();
//            info[1] = R4CommLib.rpcMicroGetCodeTimestamp().split("\n")[0];
//            byte at = R4CommLib.rpcRoGetActivationType();
//            if (at == 1) {
//                info[2] = "Immediate";
//            } else if (at == 1) {
//                info[2] = "Software";
//            } else if (at == 4) {
//                info[2] = "Hardware";
//            } else {
//                info[2] += "UNKNOWN";
//            }
//            info[3] = Integer.toString(R4CommLib.rpcRoGetDefault());
//            info[4] = Integer.toString(R4CommLib.rpcRoGetSelected());
//            info[5] = R4CommLib.rpcSysGetRepertoireName();
//            gui.showText("Info-Array constructed");
//            String serverOut = "Transfering info";
//            for (String output : info) {
//                serverOut += ";" + output;
//            }
//            return serverOut;
//        } catch (AbstractException ex) {
//            return catchedAbstractException(ex);
//        }
    String string;
        try {
            string = Integer.toString(R4CommLib.rpcRoGetSelected());
        } catch (AbstractException ex) {
            return catchedAbstractException(ex);
        }
    return "Transfering info;" + string;
    }

    /**
     * Sends an array of running orders of the SLM to the client
     *
     * @param out String-output-Stream of the server
     */
    @Override
    public String getRoList() {
        //out.println("Try to transfer running orders");
        try {
            int len = R4CommLib.rpcRoGetCount();
            String[] ros = new String[len];
            for (int i = 0; i < len; i++) {
                ros[i] = R4CommLib.rpcRoGetName(i);
            }
            gui.showText("RoList-Array constructed");
            /*
            String serverOut = "Transfering rolist";
            for (String output : ros) {
                serverOut += ";" + output;
            }
            */
            return Tool.encodeArray("Transfering rolist", ros);
        } catch (AbstractException ex) {
            return catchedAbstractException(ex);
        }
    }

    /**
     * Reboots the SLM <br>
     * not really recommended to use this
     *
     * @param out String-output-Stream of the server
     */
    @Override
    public String rebootSlm() {
        //out.println("Try to reboot the SLM");
        try {
            R4CommLib.rpcSysReboot();
            gui.showText("Flcos is rebooting");
            return "Reboot of the SLM. This may takes more than 20 seconds";
        } catch (AbstractException ex) {
            return catchedAbstractException(ex);
        }
    }

    /**
     * Opens the connection between Server and SLM
     *
     * @param out String-output-Stream of the server
     */
    @Override
    public String connectSlm() {
        //out.println("Try to connect to the SLM");
        try {
            String[] devEnumerateWinUSB = CommLib.devEnumerateWinUSB(R4CommLib.R4_WINUSB_GUID); //need to test the other funktion
            try {
                String devPath = devEnumerateWinUSB[0].split(":")[0];
                CommLib.devOpenWinUSB(devPath, 1000);
                gui.showText("Connection to the Flcos opened.");
                return "Connected to the Flcos";
            } catch (ArrayIndexOutOfBoundsException e) {
                throw new CommException("No device found", 7);
            }
        } catch (AbstractException ex) {
            return catchedAbstractException(ex);
        }
    }

    /**
     * Closes the connection between Server and SLM
     *
     * @param out String-output-Stream of the server
     */
    @Override
    public String disconnectSlm() {
        //out.println("Try to disconnect from the SLM");
        try {
            CommLib.devClose();
            gui.showText("Connection to the Flcos closed.");
            return "Disconnected from the SLM";
        } catch (AbstractException ex) {
            return catchedAbstractException(ex);
        }
    }
}
