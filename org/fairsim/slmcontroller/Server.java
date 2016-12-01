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

package org.fairsim.slmcontroller;

import com.forthdd.commlib.exceptions.*;
import com.forthdd.commlib.r4.R4CommLib;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.*;
import java.util.NoSuchElementException;
import java.util.Scanner;

/**
 * Class for the Server in the communikation chain <br>
 * SLM - Server - Client - GUI
 *
 * @author m.lachetta
 */
public class Server {

    /**
     * passes non chatched exeptions to the client
     *
     * @param ex thrown Exception
     * @param out String-output-Stream of the server
     */
    private static void catchedAbstractException(AbstractException ex, PrintWriter out) {
        out.println("Error: " + ex.getMessage() + "  ;  Code: " + ex.getCode() + "  ;  " + ex.getClass());
    }

    /**
     * sets a new selected running order
     *
     * @param ro new running order
     * @param out String-output-Stream of the server
     */
    private static void setRo(int ro, PrintWriter out, ServerGui gui) {
        out.println("Try to set running order to: " + ro);
        try {
            R4CommLib.rpcRoDeactivate();
            R4CommLib.rpcRoSetSelected(ro);
            gui.showText("Selected running order '" + ro + "'");
            out.println("Running order was set to: " + ro);
        } catch (AbstractException ex) {
            catchedAbstractException(ex, out);
        }
    }

    /**
     * activates the selected running order
     *
     * @param out String-output-Stream of the server
     */
    private static void activateRo(PrintWriter out, ServerGui gui) {
        out.println("Try to activate selected running order");
        try {
            R4CommLib.rpcRoActivate();
            gui.showText("Activated running order '" + R4CommLib.rpcRoGetSelected() + "'");
            out.println("Selected running order [" + R4CommLib.rpcRoGetSelected() + "] got activated");
        } catch (AbstractException ex) {
            catchedAbstractException(ex, out);
        }
    }

    /**
     * deactivates the current running order
     *
     * @param out String-output-Stream of the server
     */
    private static void deactivateRo(PrintWriter out, ServerGui gui) {
        out.println("Try to deactivate current running order");
        try {
            R4CommLib.rpcRoDeactivate();
            gui.showText("Deactivated current running order");
            out.println("Current running order got deactivated");
        } catch (AbstractException ex) {
            catchedAbstractException(ex, out);
        }
    }

    /**
     * Sends an array of information about the SLM to the client
     *
     * @param out String-output-Stream of the server
     */
    private static void getSlmInfo(PrintWriter out, ServerGui gui) {
        out.println("Try to transfer Slm Information");
        try {
            String[] info = new String[6];
            info[0] = R4CommLib.libGetVersion();
            info[1] = "Timestamp: " + R4CommLib.rpcMicroGetCodeTimestamp();
            byte at = R4CommLib.rpcRoGetActivationType();
            if (at == 1) {
                info[2] = "Immediate";
            } else if (at == 1) {
                info[2] = "Software";
            } else if (at == 4) {
                info[2] = "Hardware";
            } else {
                info[2] += "UNKNOWN";
            }
            info[3] = Integer.toString(R4CommLib.rpcRoGetDefault());
            info[4] = Integer.toString(R4CommLib.rpcRoGetSelected());
            info[5] = R4CommLib.rpcSysGetRepertoireName();
            gui.showText("Info-Array constructed");
            out.println("Transfering info: " + info.length);
            for (String output : info) {
                out.println(output);
            }
        } catch (AbstractException ex) {
            catchedAbstractException(ex, out);
        }
    }

    /**
     * Sends an array of running orders of the SLM to the client
     *
     * @param out String-output-Stream of the server
     */
    private static void getRoList(PrintWriter out, ServerGui gui) {
        out.println("Try to transfer running orders");
        try {
            int len = R4CommLib.rpcRoGetCount();
            String[] ros = new String[len];
            for (int i = 0; i < len; i++) {
                ros[i] = R4CommLib.rpcRoGetName(i);
            }
            gui.showText("RoList-Array constructed");
            out.println("Transfering rolist: " + ros.length);
            for (String output : ros) {
                out.println(output);
            }
        } catch (AbstractException ex) {
            catchedAbstractException(ex, out);
        }
    }

    /**
     * Reboots the SLM <br>
     * net realy recoment to use this
     *
     * @param out String-output-Stream of the server
     */
    private static void rebootSlm(PrintWriter out, ServerGui gui) {
        out.println("Try to reboot the SLM");
        try {
            R4CommLib.rpcSysReboot();
            gui.showText("SLM is rebooting");
            out.println("Reboot of the SLM. This may takes more than 10 seconds");
        } catch (AbstractException ex) {
            catchedAbstractException(ex, out);
        }
    }

    /**
     * Opens the connecten betwen Server and SLM
     *
     * @param out String-output-Stream of the server
     */
    private static void connectSlm(PrintWriter out, ServerGui gui) {
        out.println("Try to connect to the SLM");
        try {
            String[] devEnumerateWinUSB = R4CommLib.devEnumerateWinUSB(R4CommLib.R4_WINUSB_GUID); //need to test the other funktion
            try {
                String devPath = devEnumerateWinUSB[0].split(":")[0];
                R4CommLib.devOpenWinUSB(devPath, 1000);
                gui.showText("Connection to the SLM opened.");
                out.println("Connected to the SLM");
            } catch (ArrayIndexOutOfBoundsException e) {
                throw new CommException("No device found", 7);
            }
        } catch (AbstractException ex) {
            catchedAbstractException(ex, out);
        }
    }

    /**
     * Closes the connecten betwen Server and SLM
     *
     * @param out String-output-Stream of the server
     */
    private static void disconnectSlm(PrintWriter out, ServerGui gui) {
        out.println("Try to disconnect from the SLM");
        try {
            R4CommLib.devClose();
            gui.showText("Connection to the SLM closed.");
            out.println("Disconnected from the SLM");
        } catch (AbstractException ex) {
            catchedAbstractException(ex, out);
        }
    }

    /**
     * Handle what the do while Server and Client are connected to eachother
     *
     * @param client Client in the communikation chain <br>
     * SLM - Server - Client - GUI
     * @throws IOException
     */
    private static void handleConnection(Socket client, ServerGui gui) throws IOException {
        Scanner in = new Scanner(client.getInputStream());
        PrintWriter out = new PrintWriter(client.getOutputStream(), true);
        String input;
        int ro;
        while (true) {
            try {
                input = in.nextLine();
                out.println("Command '" + input + "' successfully transmitted to the server.");
            } catch (NoSuchElementException e) {
                break;
            }
            try {
                ro = Integer.parseInt(input);
                setRo(ro, out, gui);

            } catch (NumberFormatException e) {
                if (input.equals("activate")) {
                    activateRo(out, gui);
                } else if (input.equals("deactivate")) {
                    deactivateRo(out, gui);
                } else if (input.equals("info")) {
                    getSlmInfo(out, gui);
                } else if (input.equals("rolist")) {
                    getRoList(out, gui);
                } else if (input.equals("reboot")) {
                    rebootSlm(out, gui);
                } else if (input.equals("connect")) {
                    connectSlm(out, gui);
                } else if (input.equals("disconnect")) {
                    disconnectSlm(out, gui);
                } else {
                    out.println("Server do not know what to do with '" + input + "'");
                }
            }
        }

    }

    static void startServer(ServerGui gui) {
        int port = 32322;
        new Thread(new Runnable() {
            @Override
            public void run() {
                ServerSocket server;
                try {
                    server = new ServerSocket(port);
                    while (true) {
                        Socket client = null;
                        try {
                            //System.out.println("Waiting for connection from client...");
                            gui.showText("Waiting for connection from client...");
                            client = server.accept();
                            //System.out.println("Connected with: " + client.getInetAddress() + ":" + client.getLocalPort());
                            gui.showText("Connected with: " + client.getInetAddress() + ":" + client.getLocalPort());
                            handleConnection(client, gui);
                        } catch (IOException e) {
                            //System.err.println("This should not have happened 1");
                            gui.showText("This should not have happened 1");
                        } finally {
                            if (client != null) {
                                client.close();
                                try {
                                    R4CommLib.rpcRoDeactivate();
                                    R4CommLib.devClose();
                                    //System.out.println("Connection closed");
                                    gui.showText("Connection closed");
                                } catch (AbstractException e) {
                                    //System.err.println("This should not have happened 2");
                                    gui.showText("This should not have happened 2");
                                }
                            }
                        }
                    }
                } catch (IOException ex) {
                    //System.err.println("This should not have happened 3");
                    gui.showText("This should not have happened 3");
                }
            }
        }).start();
    }

}
