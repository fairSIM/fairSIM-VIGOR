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
import java.net.ServerSocket;
import java.net.Socket;
import java.util.NoSuchElementException;
import java.util.Scanner;

/**
 * Abstract class for a server for network communication in an extra thread,
 * build for a single connected client
 * @author m.lachetta
 */
public abstract class AbstractServer extends Thread {

    protected ServerGui gui;
    private final int port;
    private final ServerSocket server;
    private Socket client;
    protected Scanner in;
    protected PrintWriter out;
    public boolean interrupted;

    /**
     * creates a new server
     * @param gui gui for this server
     * @throws IOException if creating went wrong
     */
    protected AbstractServer(ServerGui gui) throws IOException {
        this.gui = gui;
        port = 32322;
        server = new ServerSocket(port);
        interrupted = false;
    }

    /**
     * method that is called after the client was connected to this server
     */
    protected abstract void buildUpConnection();

    /**
     * method that is called after the connection to the client was closed
     */
    protected abstract void buildDownConnection();

    /**
     * method that is called after this server received a command from the client
     * @param command command from the client
     * @return answer from this server
     */
    protected abstract String handleCommand(String command);

    /**
     * receives commands and sends answers to the client
     * @throws IOException 
     */
    private void handleConnection() throws IOException {
        String input;
        try {
            while (!interrupted) {
                input = in.nextLine();
                //out.println("Server: Command '" + input + "' successfully transmitted to the server.");
                out.println(handleCommand(input));
            }
        } catch (NoSuchElementException ex) {
        }
    }
    
    /**
     * closes the connection to the client
     */
    public final void close() {
        new Thread(new Runnable() {
            public void run() {
                interrupted = true;
                try {
                    server.close();
                } catch (Exception ex) {
                    System.err.println(ex);
                }
            }
        }).start();
    }

    /**
     * waits for - and handles connections to the client
     */
    @Override
    public final void run() {
        while (!interrupted) {
            client = null;
            in = null;
            out = null;
            try {
                gui.showText("Waiting for connection from client...");
                client = server.accept();
                gui.showText("Connected with: " + client.getInetAddress() + ":" + client.getLocalPort());
                buildUpConnection();
                in = new Scanner(client.getInputStream());
                out = new PrintWriter(client.getOutputStream(), true);
                handleConnection();
            } catch (IOException ex) {
                gui.showText(ex.toString());
            } finally {
                if (client != null) {
                    buildDownConnection();
                    try {
                        client.close();
                    } catch (IOException ex) {
                        gui.showText(ex.toString());
                    }
                    gui.showText("Connection closed");
                }
            }
        }
    }
    
    /**
     * interface for a server gui
     */
    public interface ServerGui {
        /**
         * shows a message at the gui
         * @param message massage to be shown
         */
        void showText(String message);
    }
}
