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

import java.net.*;
import java.io.*;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Class fo the Client in the communikation chain <br>
 * SLM - Server - Client - GUI
 *
 * @author m.lachetta
 */
public class Client {

    private final SlmPanel slmGui;
    private final Socket server;
    private final Scanner in;
    private final PrintWriter out;
    String[] info;
    String[] roList;
    String output;
    BlockingQueue<Instruction> instructions;

    /**
     * Constructor for the Client
     *
     * @param adress IP/Name of the host-server
     * @param port for the Connection
     * @param slmGui Gui for the SLM
     * @throws IOException
     */
    private Client(String adress, int port, SlmPanel slmGui) throws IOException {
        this.slmGui = slmGui;
        slmGui.showText("Trying to connect to: " + adress + ":" + port);
        server = new Socket(adress, port);
        slmGui.showText("Connected to: " + adress + ":" + port);
        in = new Scanner(server.getInputStream());
        out = new PrintWriter(server.getOutputStream(), true);
        instructions = new LinkedBlockingQueue<>();
    }

    /**
     * closes the Connection to the host-server
     */
    private void disconnect() {
        if (server != null) {
            try {
                server.close();
            } catch (IOException e) {
            }
        }

    }

    /**
     * Reciving an array from the host-server
     *
     * @param output optput from the server
     * @return the recived array
     */
    private String[] receivingData(String output) {
        String temp = output.split(": ")[1];
        int len = Integer.parseInt(temp);
        String[] data = new String[len];
        for (int i = 0; i < len; i++) {
            data[i] = in.nextLine();
            if (data[i].startsWith("Timestamp")) { //correct empty line after Timestamp
                in.nextLine();
            }
        }
        return data;
    }

    /**
     * handle the answers of the host-server
     *
     * @param input first answer from the server
     */
    private void handleAction(String input) {

        out.println(input);
        output = in.nextLine();
        //slmGui.showText(output); //Command 'input' successfully transmitted to the server
        output = in.nextLine();
        //slmGui.showText(output);
        if (output.startsWith("Try ")) {
            output = in.nextLine();
            if (output.startsWith("Transfering info: ")) {
                info = receivingData(output);
            } else if (output.startsWith("Transfering rolist: ")) {
                roList = receivingData(output);
            } else {
                slmGui.showText(output);
            }
        }
    }

    /**
     * Starts the Client, used from the SlmPanel.java
     *
     * @param serverAdress IP/name of the host-server
     * @param slmGui Gui for the SLM
     */
    static void startClient(String adress, int port, SlmPanel slmGui) {
        slmGui.setConnectionLabel(adress, port);
        Thread connection = new Thread(new Runnable() {
            @Override
            public void run() {
                Client client = null;
                while (true) {
                    try {
                        client = new Client(adress, port, slmGui);
                        slmGui.registerClient(client);
                        Instruction input;
                        while (true) {
                            input = client.instructions.take();
                            input.lock.lock();
                            try {
                                client.handleAction(input.command);
                            } finally {
                                input.condition.signal();
                                input.lock.unlock();
                            }
                        }
                    } catch (UnknownHostException e) {
                        slmGui.showText("Error: UnknownHostException");
                    } catch (ConnectException e) {
                        slmGui.showText("Error: Connection failed to : " + adress + ":" + port);
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException ex) {
                            slmGui.showText("Error: InterruptedException 1");
                        }
                    } catch (IOException e) {
                        slmGui.showText("Error: IOException");
                    } catch (NoSuchElementException e) {
                        slmGui.showText("Error: Connection lost to: " + adress + ":" + port);
                    } catch (InterruptedException ex) {
                        slmGui.showText("Error: InterruptedException 2");
                    } finally {
                        slmGui.unregisterClient();
                        if (client != null) {
                            client.disconnect();
                        }
                    }
                }
            }
        });

        connection.start();
    }

}
