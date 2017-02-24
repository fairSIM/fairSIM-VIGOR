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
 * Class of the Client in the communikation chain <br>
 * SLM - Server - Client - GUI
 *
 * @author m.lachetta
 */
public class ControllerClient extends AbstractClient {
    /*
    String serverAdress;
    int serverPort;
    private final ClientGui clientGui;
    private Socket serverSocket;
    private Scanner in;
    private PrintWriter out;
    String output;
    BlockingQueue<Instruction> instructions;
    */
    ControllerClientGui clientGui;
    String[] slmInfo;
    String[] slmList;

    /**
     * Constructor for the Client
     *
     * @param serverAdress IP/Name of the host-server
     * @param serverPort for the Connection
     * @param clientGui Gui for the SLM
     */
    protected ControllerClient(String serverAdress, int serverPort, ControllerClientGui clientGui) {
        super(serverAdress, serverPort, clientGui);
        this.clientGui = clientGui;
        /*
        clientGui.showText("Client: Trying to connect to: " + serverAdress + ":" + serverPort);
        serverSocket = new Socket(serverAdress, serverPort);
        clientGui.showText("Client: Connected to: " + serverAdress + ":" + serverPort);
        in = new Scanner(serverSocket.getInputStream());
        out = new PrintWriter(serverSocket.getOutputStream(), true);
        */
        
    }
    /*
    protected void connectToServer() throws IOException  {
        clientGui.showText("Client: Trying to connect to: " + serverAdress + ":" + serverPort);
        serverSocket = new Socket(serverAdress, serverPort);
        clientGui.showText("Client: Connected to: " + serverAdress + ":" + serverPort);
        in = new Scanner(serverSocket.getInputStream());
        out = new PrintWriter(serverSocket.getOutputStream(), true);
    }
    */
    /**
     * closes the Connection to the host-server
     */
    /*
    protected void disconnect() {
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException e) {
            }
        }

    }
    */
    /**
     * Reciving an array from the host-server
     *
     * @param output optput from the server
     * @return the recived array
     */
    /*
    private String[] receivingData(String output) {
        String[] split = output.split(";");
        String[] data = new String[split.length - 1];
        for (int i = 0; i < data.length; i++) {
            data[i] = split[i + 1];
        }
        return data;
    }
    */

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
        } else if (answer.startsWith("Slm: Error: ")) {
            clientGui.handleSlmError(answer);
        } else if (answer.startsWith("Arduino: Error: ")) {
            clientGui.handleArduinoError(answer);
        } else {
            clientGui.showText(answer);
        }
    }
    
    /**
     * Starts the Client, used from the SlmPanel.java
     *
     * @param serverAdress IP/name of the host-server
     * @param clientGui Gui for the SLM
     */
    /*
    static void startClient(final String adress, final int port, final ClientGui clientGui) {
        clientGui.setConnectionLabel(adress, port);
        ControllerClient client = new ControllerClient(adress, port, clientGui);
        Thread connection = new Thread(new Runnable() {
            @Override
            public void run() {
                while (true) {
                    try {
                        client.connectToServer();
                        clientGui.registerClient(client);
                        Instruction input;
                        while (true) {
                            System.out.println(client.serverAdress);
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
    */
}
