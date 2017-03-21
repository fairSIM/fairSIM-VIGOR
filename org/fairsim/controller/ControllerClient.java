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

import org.fairsim.utils.Tool;

/**
 * Class of the Client in the communication chain <br>
 * SLM - Server - Client - GUI
 *
 * @author m.lachetta
 */
public class ControllerClient extends AbstractClient {
    String[] slmInfo;
    String[] slmList;
    ArduinoRunningOrder[] arduinoRos;

    /**
     * Constructor for the Client
     *
     * @param serverAdress IP/Name of the host-server
     * @param serverPort for the Connection
     * @param clientGui Gui for the SLM
     */
    protected ControllerClient(String serverAdress, int serverPort, AdvancedGui.ClientGui controllerPanel) {
        super(serverAdress, serverPort, controllerPanel);
    }
    
    /**
     * handle the answers of the host-server
     *
     * @param answer The answer from the server
     */
    @Override
    protected void handleServerAnswer(String answer) {
        //clientGui.showText(output);
        if (answer.startsWith("Slm: Transfering info")) {
            slmInfo = Tool.decodeArray(answer);
        } else if (answer.startsWith("Slm: Transfering rolist")) {
            slmList = Tool.decodeArray(answer);
        } else if (answer.startsWith("Arduino: Transfering rolist")) {
            String[] stringRos = Tool.decodeArray(answer);
            int len = stringRos.length;
            arduinoRos = new ArduinoRunningOrder[len];
            for (int i = 0; i < len; i++) {
                arduinoRos[i] = new ArduinoRunningOrder(stringRos[i]);
            }
        } else if (answer.startsWith("Slm: Error: ") || answer.startsWith("Arduino: Error: ")) {
            gui.handleError(answer);
        } else {
            gui.showText(answer);
        }
    }

    @Override
    protected void handleTimeout(String command) {
        gui.showText("Timeout for command: " + command);
        gui.interruptInstruction();
    }

    @Override
    protected void handleInterrupt(String command) {
        gui.showText("Interrupt for command: " + command);
        gui.interruptInstruction();
    }
    
    class ArduinoRunningOrder {
        
        String name;
        int syncDelay;
        int syncFreq;
        int exposureTime;
        
        ArduinoRunningOrder(String encodedRo) {
            String[] stringArray = encodedRo.split(",");
            name = stringArray[0];
            syncDelay = Integer.parseInt(stringArray[1]);
            syncFreq = Integer.parseInt(stringArray[2]);
            exposureTime = Integer.parseInt(stringArray[3]);
        }
    }
}
