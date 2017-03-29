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
 * Class for the controller client
 *
 * @author m.lachetta
 */
public class ControllerClient extends AbstractClient {
    String[] deviceInfo;
    String[] deviceList;
    ArduinoRunningOrder[] arduinoRos;

    /**
     * Constructor for this
     *
     * @param serverAdress IP/Name of the server
     * @param serverPort port of the server
     * @param clientGui gui for this
     */
    protected ControllerClient(String serverAdress, int serverPort, AbstractClient.ClientGui controllerPanel) {
        super(serverAdress, serverPort, controllerPanel);
    }
    
    /**
     * handle the answers of the host-server
     *
     * @param answer The answer from the server
     */
    @Override
    protected void handleServerAnswer(String answer) {
        if (answer.startsWith("Slm: Transfering info")) {
            // updates the information of the device
            deviceInfo = Tool.decodeArray(answer);
        } else if (answer.startsWith("Slm: Transfering rolist")) {
            // updates running orders of the device
            deviceList = Tool.decodeArray(answer);
        } else if (answer.startsWith("Arduino: Transfering rolist")) {
            // updates running orders of the arduino
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
    
    /**
     * class for running orders of the arduino
     */
    class ArduinoRunningOrder {
        String name;
        int syncDelay;
        int syncFreq;
        int exposureTime;
        
        /**
         * constructs from an encoded arduino running order an new
         * @param encodedRo the encoded running order
         */
        ArduinoRunningOrder(String encodedRo) {
            String[] stringArray = encodedRo.split(",");
            name = stringArray[0];
            syncDelay = Integer.parseInt(stringArray[1]);
            syncFreq = Integer.parseInt(stringArray[2]);
            exposureTime = Integer.parseInt(stringArray[3]);
        }
    }
}
