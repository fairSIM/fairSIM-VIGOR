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

package org.fairsim.transport;

import java.net.DatagramSocket;
import java.net.DatagramPacket;
import java.net.Socket;
import java.net.SocketException;
import java.net.SocketTimeoutException;


import java.nio.ByteBuffer;
import java.io.IOException;

import org.fairsim.utils.Tool;

/** Control program flow thorugh UDP packets */
public class ControlConnection {
    
    private static final int maxCommandLength = 128;
    private UdpListener listenThread = null;

    /** Implement to get notified whenever packets arrive */
    public interface Notify {
	public void message( String msg );
    }

    /** Call to block program flow until a packet arrives */
    public interface Wait {
	public void block(long timeoutMillis);
	public void block();
    }



    /** Send a command */
    static void sendCommand(String command, String hostname, int port ) 
	throws java.net.UnknownHostException, java.io.IOException  { 
	
	// get the command, as bytes
	byte [] cmdBytes = command.getBytes("US-ASCII");

	if (cmdBytes.length >= maxCommandLength)
	    throw new java.io.IOException("Command too long");

	// get a datagram socket
	DatagramSocket socket = new DatagramSocket();

	DatagramPacket sendpacket = 
		new DatagramPacket( cmdBytes, cmdBytes.length,
			java.net.InetAddress.getByName(hostname), port);
	socket.send(sendpacket);
    }

    
    /** Creates a CommandConnection listening on a given port. */
    public ControlConnection( int port ) throws java.net.SocketException {

	UdpListener udpListen = new UdpListener(port);
	udpListen.start();

    }







    /** Thread listening for UDP packets */
    private class UdpListener extends Thread {

	final DatagramSocket listenSocket;
	boolean haltThread = false;

	UdpListener( int port ) throws SocketException {
	    listenSocket = new DatagramSocket(port);
	    listenSocket.setSoTimeout(500);
	}

	@Override
	public void run() {
	    	    
	    while (!haltThread) {

		byte [] buf		    = new byte[maxCommandLength];
		ByteBuffer bbr	    = ByteBuffer.wrap(buf);
		DatagramPacket packet   = new DatagramPacket(buf, buf.length);
		bbr.order( java.nio.ByteOrder.LITTLE_ENDIAN );

		try {
		    listenSocket.receive(packet);
		    String  retState = getStringFromBytes(buf, 0, maxCommandLength);
		    Tool.trace( retState );
		}
		catch (SocketTimeoutException e) {
		    // ignore
		}
		catch (IOException e ) {
		    Tool.error("Command UDP exception: "+e,false);
		}
	    }
        
	    listenSocket.close();
    	}
    }



    /** Converts null-terminated byte array to a string */
    static String getStringFromBytes( byte [] in, int off, int maxLen ) {

	int pos = 0;
	for (int i=off; i<maxLen+off; i++) {
	    pos = i;
	    if (in[i] ==0) break;
	}
	
	String ret = new String( in , off, pos-off );
	return ret;

    }


    public static void main( String [] arg ) {

	if (arg.length<1) {
	    System.out.println("Usage: [send/listen] hostname message");
	    return;
	}

	// listen for UDP packets
	if (arg[0].equals("listen")) {
	    
	    try {
		ControlConnection cc = new ControlConnection(32320);
		Thread.sleep(3600*1000);
	    } catch ( Exception e ) {
		throw new RuntimeException(e);
	    }
	}

	// send UDP packets
	if (arg[0].equals("send")) {
	    if (arg.length < 3) {
		System.out.println("Missing: host and/or message");
		return;
	    }

	    try {
		ControlConnection.sendCommand(arg[2], arg[1], 32320);
	    } catch (Exception e) {
		throw new RuntimeException(e);
	    }

	}



    }




}
