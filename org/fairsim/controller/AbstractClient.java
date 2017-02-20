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
import java.net.ConnectException;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 *
 * @author m.lachetta
 */
public abstract class AbstractClient extends Thread {

    protected String serverAdress;
    protected int serverPort;
    protected ClientGui clientGui;
    protected Socket serverSocket;
    protected Scanner in;
    protected PrintWriter out;
    protected String output;
    protected BlockingQueue<Instruction> instructions;

    protected AbstractClient(String serverAdress, int serverPort, ClientGui clientGui) {
        this.serverAdress = serverAdress;
        this.serverPort = serverPort;
        this.clientGui = clientGui;
        instructions = new LinkedBlockingQueue<>();
    }

    private void connectToServer() throws IOException {
        clientGui.showText("Client: Trying to connect to: " + serverAdress + ":" + serverPort);
        serverSocket = new Socket(serverAdress, serverPort);
        clientGui.showText("Client: Connected to: " + serverAdress + ":" + serverPort);
        in = new Scanner(serverSocket.getInputStream());
        out = new PrintWriter(serverSocket.getOutputStream(), true);
    }

    private void disconnect() {
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException e) {
            }
        }
    }

    protected abstract void handleAction(String input);

    @Override
    public void run() {
        while (true) {
            try {
                connectToServer();
                clientGui.registerClient(this);
                Instruction input;
                while (true) {
                    input = instructions.take();
                    input.lock.lock();
                    try {
                        handleAction(input.command);
                    } finally {
                        input.condition.signal();
                        input.lock.unlock();
                    }
                }
            } catch (UnknownHostException e) {
                clientGui.showText("Client: Error: UnknownHostException");
            } catch (ConnectException e) {
                clientGui.showText("Client: Error: Connection failed to : " + serverAdress + ":" + serverPort);
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException ex) {
                    clientGui.showText("Client: Error: InterruptedException 1");
                }
            } catch (IOException e) {
                clientGui.showText("Client: Error: IOException");
            } catch (NoSuchElementException e) {
                clientGui.showText("Client: Error: Connection lost to: " + serverAdress + ":" + serverPort);
            } catch (InterruptedException ex) {
                clientGui.showText("Client: Error: InterruptedException 2");
            } finally {
                clientGui.unregisterClient(this);
                if (this != null) {
                    disconnect();
                }
            }
        }
    }
}
