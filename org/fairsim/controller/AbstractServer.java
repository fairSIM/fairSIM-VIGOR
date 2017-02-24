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
 *
 * @author m.lachetta
 */
public abstract class AbstractServer extends Thread{
    protected ServerGui gui;
    private int port;
    private ServerSocket server;
    private Socket client;
    protected Scanner in;
    protected PrintWriter out;
    
    protected AbstractServer(ServerGui gui) throws IOException {
        this.gui = gui;
        port = 32322;
        server = new ServerSocket(port);
    }

    protected abstract void buildUpConnection();
    protected abstract void buildDownConnection();
    protected abstract String handleCommand(String input);
    
    private void handleConnection() throws IOException {
        String input;
        while (true) {
            try {
                input = in.nextLine();
                out.println("Server: Command '" + input + "' successfully transmitted to the server.");
            } catch (NoSuchElementException e) {
                break;
            }
            out.println( handleCommand(input) );
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
}