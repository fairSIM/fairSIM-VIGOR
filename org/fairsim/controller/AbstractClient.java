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
import java.util.concurrent.TimeUnit;

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
    protected final int TIMEOUT = 5;
    private String output;
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
    
    protected void addInstruction(String command) {
        Instruction instruction = new Instruction(command);
        instruction.lock.lock();
        try {
            instructions.add(instruction);
            instruction.condition.await(TIMEOUT, TimeUnit.SECONDS);
        } catch (InterruptedException ex) {
            clientGui.showText("Gui: Error: Instruction timed out: " + ex.toString());
        } finally {
            instruction.lock.unlock();
        }
    }

    protected abstract void handleServerAnswer(String answer);

    @Override
    public void run() {
        while (!isInterrupted()) {
            try {
                connectToServer();
                Instruction input;
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        clientGui.registerClient(AbstractClient.this);
                    }
                }).start();
                while (!isInterrupted()) {
                    input = instructions.take();
                    input.lock.lock();
                    try {
                        out.println(input.command);
                        output = in.nextLine();
                        //clientGui.showText(output); //Command 'input' successfully transmitted to the server
                        output = in.nextLine();
                        handleServerAnswer(output);
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
                    interrupt();
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
