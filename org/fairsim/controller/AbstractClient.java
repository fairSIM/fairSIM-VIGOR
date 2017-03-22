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
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 *
 * @author m.lachetta
 */
public abstract class AbstractClient extends Thread {

    protected final String serverAdress;
    protected final int serverPort;
    protected final AdvancedGui.ClientGui gui;
    protected Socket serverSocket;
    protected Scanner in;
    protected PrintWriter out;
    protected static final int TIMEOUT = 1000;
    private String output;
    protected final BlockingQueue<Instruction> instructions;

    protected AbstractClient(String serverAdress, int serverPort, AdvancedGui.ClientGui gui) {
        this.serverAdress = serverAdress;
        this.serverPort = serverPort;
        this.gui = gui;
        instructions = new LinkedBlockingQueue<>();
    }

    private void connectToServer() throws IOException {
        gui.showText("Client: Trying to connect to: " + serverAdress + ":" + serverPort);
        serverSocket = new Socket(serverAdress, serverPort);
        gui.showText("Client: Connected to: " + serverAdress + ":" + serverPort);
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
    
    protected final void addInstruction(String command) {
        Instruction instruction = new Instruction(command);
        instruction.lock.lock();
        try {
            instructions.add(instruction);
            boolean outtimed = !instruction.condition.await(TIMEOUT, TimeUnit.MILLISECONDS);
            if (outtimed) {
                handleTimeout(instruction.command);
            }
        } catch (InterruptedException ex) {
            handleInterrupt(instruction.command);
        } finally {
            instruction.lock.unlock();
        }
    }

    protected abstract void handleServerAnswer(String answer);
    protected abstract void handleTimeout(String command);
    protected abstract void handleInterrupt(String command);

    @Override
    public final void run() {
        while (!isInterrupted()) {
            try {
                connectToServer();
                Instruction input;
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        gui.registerClient();
                    }
                }).start();
                while (!isInterrupted()) {
                    input = instructions.take();
                    input.lock.lock();
                    try {
                        out.println(input.command);
                        //output = in.nextLine();
                        //clientGui.showText(output); //Command 'input' successfully transmitted to the server
                        output = in.nextLine();
                        handleServerAnswer(output);
                    } finally {
                        input.condition.signal();
                        input.lock.unlock();
                    }
                }
            } catch (IOException ex) {
                gui.showText(ex + ";\t in: " + getClass());
                try {
                    Thread.sleep(2000);
                } catch (InterruptedException e) {
                    interrupt();
                    gui.showText(ex + ";\t in: " + getClass());
                }
            } catch (NoSuchElementException ex) {
                gui.showText(getClass() + ": Connection lost to: " + serverAdress + ":" + serverPort);
            } catch (InterruptedException ex) {
                gui.showText(ex + ";\t in: " + getClass());
            } finally {
                gui.unregisterClient();
                disconnect();
            }
        }
    }
    
    private final class Instruction {
        Lock lock;
        Condition condition;
        String command;
        Instruction(String command) {
            lock = new ReentrantLock();
            condition = lock.newCondition();
            this.command = command;
        }
    }
}
