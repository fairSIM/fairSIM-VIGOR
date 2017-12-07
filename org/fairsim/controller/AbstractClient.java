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
import java.net.Socket;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Abstract class for a client for network communication in an extra thread
 *
 * @author m.lachetta
 */
public abstract class AbstractClient extends Thread {

    protected final String serverAdress;
    protected final int serverPort;
    protected final ClientGui gui;
    protected Socket serverSocket;
    protected Scanner in;
    protected PrintWriter out;
    protected static final int TIMEOUT = 2500;
    private String output;
    protected final BlockingQueue<Instruction> instructions;

    /**
     * creates a new client
     *
     * @param serverAdress ip of the server
     * @param serverPort port for the communication
     * @param gui gui for this client
     */
    protected AbstractClient(String serverAdress, int serverPort, ClientGui gui) {
        this.serverAdress = serverAdress;
        this.serverPort = serverPort;
        this.gui = gui;
        instructions = new LinkedBlockingQueue<>();
    }

    /**
     * connects this client to the server
     *
     * @throws IOException if connecting went wrong
     */
    private void connectToServer() throws IOException {
        gui.showText("Client: Trying to connect to: " + serverAdress + ":" + serverPort);
        serverSocket = new Socket(serverAdress, serverPort);
        gui.showText("Client: Connected to: " + serverAdress + ":" + serverPort);
        in = new Scanner(serverSocket.getInputStream());
        out = new PrintWriter(serverSocket.getOutputStream(), true);
    }

    /**
     * disconnects this client from the server
     */
    private void disconnect() {
        if (serverSocket != null) {
            try {
                serverSocket.close();
            } catch (IOException e) {
            }
        }
    }

    /**
     * Adds an instruction to the queue which includes the commands to be send
     * to the server
     *
     * @param command command for the server
     */
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

    /**
     * method to handle the answers from the server
     *
     * @param answer answer from the server
     */
    protected abstract void handleServerAnswer(String answer);

    /**
     * method to handle a timeout for a command
     *
     * @param command command which was timed out
     */
    protected abstract void handleTimeout(String command);

    /**
     * method to handle an interrupt
     *
     * @param command
     */
    protected abstract void handleInterrupt(String command);

    /**
     * connects client to the server and reconnects if connection gets lost,
     * interrupt() closes the connection to the server
     */
    @Override
    public final void run() {
        while (!isInterrupted()) {
            try {

                // build up connection
                connectToServer();
                Instruction input;
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        gui.registerClient();
                    }
                }).start();

                //active loop while connected
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

                // exception handling for automatical reconnects
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

    /**
     * class that wraps commands for the server with a lock and a condition to
     * have commands thread save
     */
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
    
    /**
     * interface for panels in this gui
     */
    interface ClientGui {

        /**
         * shows text on the gui
         * @param text text to be shown
         */
        void showText(String text);

        /**
         * called after client was connected to the server
         */
        void registerClient();

        /**
         * called after client was disconnected from the server
         */
        void unregisterClient();

        /**
         * should handdle error answers from the server
         * @param answer the error answer
         */
        void handleError(String answer);

        /**
         * should be called if an instruction was interrupted
         */
        void interruptInstruction();
    }
}
