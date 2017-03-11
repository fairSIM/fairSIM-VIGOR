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

import java.io.IOException;
import java.io.PrintWriter;
import java.net.*;
import java.nio.charset.Charset;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.DataFormatException;
import org.fairsim.utils.Tool;

/**
 * Class for the Server in the communikation chain <br>
 * SLM - Server - Client - GUI
 *
 * @author m.lachetta
 */
public class ControllerServer extends AbstractServer {

    SlmController slm;
    ArduinoController arduino;

    /**
     * Constructor for the Server
     *
     * @param gui GUI of the Server
     * @param slm
     * @param arduino
     * @throws IOException throwen if TCP-Connection failed
     */
    private ControllerServer(ServerGui gui, SlmController slm, ArduinoController arduino) throws IOException {
        super(gui);
        this.slm = slm;
        this.arduino = arduino;
    }

    protected String handleCommand(String input) {
        String serverOut = "---";
        if (input.startsWith("slm->")) {
            input = input.split("->")[1];
            out.print("Slm: ");
            try {
                serverOut = slm.setRo(Integer.parseInt(input));

            } catch (NumberFormatException e) {
                if (input.equals("activate")) {
                    serverOut = slm.activateRo();
                } else if (input.equals("deactivate")) {
                    serverOut = slm.deactivateRo();
                } else if (input.equals("info")) {
                    serverOut = slm.getSlmInfo();
                } else if (input.equals("rolist")) {
                    serverOut = slm.getRoList();
                } else if (input.equals("reboot")) {
                    serverOut = slm.rebootSlm();
                } else if (input.equals("connect")) {
                    serverOut = slm.connectSlm();
                } else if (input.equals("disconnect")) {
                    serverOut = slm.disconnectSlm();
                } else {
                    serverOut = "Slm-Server do not know what to do with '" + input + "'";
                }
            }
        } else if (input.startsWith("arduino->")) {
            input = input.split("->")[1];
            out.print("Arduino: ");
            if (input.equals("connect")) {
                serverOut = arduino.connect();
            } else if (input.equals("disconnect")) {
                serverOut = arduino.disconnect();
            } else if (input.equals("rolist")) {
                serverOut = arduino.getRoList();
            } else if (input.startsWith("movie;")) {
                int[] movieArray = Tool.decodeIntArray(input);
                serverOut = arduino.startMovie(movieArray[0], movieArray[1]);
            } else if (input.startsWith("photo;")) {
                int runningOrder = Integer.parseInt(input.split(";")[1]);
                serverOut = arduino.takePhoto(runningOrder);
            } else if (input.length() == 1) {
                char c = input.charAt(0);
                serverOut = arduino.sendChar(c);
            } else {
                serverOut = "Arduino-Server do not know what to do with '" + input + "'";
            }
        }
        return serverOut;
    }

    /**
     * Creates and starts a new Controller-Server
     *
     * @param gui
     * @param slm
     * @param arduino
     * @return returns a ControllerServer-Object or a null-pointer if something
     * went wrong
     */
    static ControllerServer startControllerServer(ControllerServerGui gui, SlmController slm, ArduinoController arduino) {
        try {
            ControllerServer serverObject = new ControllerServer(gui, slm, arduino);
            serverObject.start();
            return serverObject;
        } catch (IOException ex) {
            return null;
        }
    }

    @Override
    protected void buildUpConnection() {
    }

    @Override
    protected void buildDownConnection() {
        gui.showText(slm.deactivateRo());
        gui.showText(slm.disconnectSlm());
        gui.showText(arduino.disconnect());
    }

}
