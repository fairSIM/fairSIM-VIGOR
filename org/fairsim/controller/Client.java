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

import java.net.*;
import java.io.*;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * Class of the Client in the communikation chain <br>
 * SLM - Server - Client - GUI
 *
 * @author m.lachetta
 */
public class Client {

    private final ClientGui clientGui;
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
    private Client(String adress, int port, ClientGui slmGui) throws IOException {
        this.clientGui = slmGui;
        slmGui.showText("Client: Trying to connect to: " + adress + ":" + port);
        server = new Socket(adress, port);
        slmGui.showText("Client: Connected to: " + adress + ":" + port);
        in = new Scanner(server.getInputStream());
        out = new PrintWriter(server.getOutputStream(), true);
        instructions = new LinkedBlockingQueue<Instruction>();
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
        /*
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
         */
        //new
        String[] split = output.split(";");
        String[] data = new String[split.length - 1];
        for (int i = 0; i < data.length; i++) {
            data[i] = split[i + 1];
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
        //clientGui.showText(output); //Command 'input' successfully transmitted to the server
        output = in.nextLine();
        //clientGui.showText(output);
        if (output.startsWith("Slm: Transfering info")) {
            info = receivingData(output);
        } else if (output.startsWith("Slm: Transfering rolist")) {
            roList = receivingData(output);
        } else {
            clientGui.showText(output);
        }
    }

    /**
     * Starts the Client, used from the SlmPanel.java
     *
     * @param serverAdress IP/name of the host-server
     * @param clientGui Gui for the SLM
     */
    static void startClient(final String adress, final int port, final ClientGui clientGui) {
        clientGui.setConnectionLabel(adress, port);
        Thread connection = new Thread(new Runnable() {
            @Override
            public void run() {
                Client client = null;
                while (true) {
                    try {
                        client = new Client(adress, port, clientGui);
                        clientGui.registerClient(client);
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
                        clientGui.showText("Client: Error: UnknownHostException");
                    } catch (ConnectException e) {
                        clientGui.showText("Client: Error: Connection failed to : " + adress + ":" + port);
                        try {
                            Thread.sleep(2000);
                        } catch (InterruptedException ex) {
                            clientGui.showText("Client: Error: InterruptedException 1");
                        }
                    } catch (IOException e) {
                        clientGui.showText("Client: Error: IOException");
                    } catch (NoSuchElementException e) {
                        clientGui.showText("Client: Error: Connection lost to: " + adress + ":" + port);
                    } catch (InterruptedException ex) {
                        clientGui.showText("Client: Error: InterruptedException 2");
                    } finally {
                        clientGui.unregisterClient();
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
