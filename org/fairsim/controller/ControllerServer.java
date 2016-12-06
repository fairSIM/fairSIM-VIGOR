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

/**
 * Class for the Server in the communikation chain <br>
 * SLM - Server - Client - GUI
 *
 * @author m.lachetta
 */
public class ControllerServer implements Runnable{

    ServerGui gui;
    SlmController slm;
    ArduinoController arduino;
    int port;
    ServerSocket server;
    Socket client;
    Scanner in;
    PrintWriter out;

    private ControllerServer(ServerGui gui, SlmController slm, ArduinoController arduino) throws IOException {
        this.gui = gui;
        this.slm = slm;
        this.arduino = arduino;
        port = 32322;
        server = new ServerSocket(port);
    }

    /**
     * Handle what the do while Server and Client are connected to eachother
     *
     * @param client Client in the communikation chain <br>
     * SLM - Server - Client - GUI
     * @throws IOException
     */
    private void handleConnection() throws IOException {
        String input;
        int ro;
        while (true) {
            try {
                input = in.nextLine();
                out.println("Server: Command '" + input + "' successfully transmitted to the server.");
            } catch (NoSuchElementException e) {
                break;
            }
            String serverOut = "---";
            if (input.startsWith("slm->")) {
                input = input.split("->")[1];
                out.print("Slm: ");
                try {
                    ro = Integer.parseInt(input);
                    serverOut = slm.setRo(ro);

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
                } else {
                    byte[] command = input.getBytes(Charset.forName("UTF-8"));
                    serverOut = arduino.sendCommandToArduino(command);
                }
            }
            out.println(serverOut);
        }

    }

    static ControllerServer startServer(ServerGui gui, SlmController slm, ArduinoController arduino) {
        try {
            ControllerServer serverObjekt = new ControllerServer(gui, slm, arduino);
            new Thread(serverObjekt).start();
            return serverObjekt;
        } catch (IOException ex) {
            return null;
        }
    }

    @Override
    public void run() {
        while (true) {
            client = null;
            in = null;
            out = null;
            try {
                gui.showText("Waiting for connection from client...");
                client = server.accept();
                gui.showText("Connected with: " + client.getInetAddress() + ":" + client.getLocalPort());
                in = new Scanner(client.getInputStream());
                out = new PrintWriter(client.getOutputStream(), true);
                handleConnection();
            } catch (IOException e) {
                gui.showText("This should not have happened 1");
            } finally {
                if (client != null) {
                    gui.showText(slm.deactivateRo());
                    gui.showText(slm.disconnectSlm());
                    gui.showText(arduino.disconnect());
                    try {
                        client.close();
                    } catch (IOException ex) {
                        gui.showText("This should not have happened 2");
                    }
                    gui.showText("Connection closed");
                }
            }
        }
    }

}
