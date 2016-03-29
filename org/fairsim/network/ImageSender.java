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

package org.fairsim.network;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;

import java.net.Socket;
import java.net.ServerSocket;
import java.net.InetAddress;

import java.util.List;
import java.util.ArrayList;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

public class ImageSender {


    // buffer for images to send
    BlockingQueue<ImageWrapper> imageQueue = new ArrayBlockingQueue<ImageWrapper>(16);
   
    // list of TCP connections to use
    List<SendingThread> connection = new ArrayList<SendingThread>();


    /** Start ImageSender without initial connection */
    public ImageSender() {

    }

    /** open connections to all hosts on default port */
    public ImageSender( String ... host ) throws java.net.UnknownHostException {
	for (String h : host )
	    connect( InetAddress.getByName( h ) , null);
    }


    /** Connect to 'host' at 'port', starts a new sender thread */
    public void connect(InetAddress addr, Integer port) {
	SendingThread s = new SendingThread( addr, port);
	s.start();
	connection.add(s);
    }

    /** Connect to 'host' at 'port', starts a new sender thread */
    public void connect(String host, Integer port) 
	throws java.net.UnknownHostException {
	SendingThread s = new SendingThread( InetAddress.getByName( host ), port);
	s.start();
	connection.add(s);
    }



    /** Non-blocking image send. Returns if the image was sucessfully
     * queued for sending. */
    public boolean queueImage( ImageWrapper img ) {
	if (img==null)
	    return false;
	return imageQueue.offer( img);
    }



    /** This thread opens a TCP socket and sends images in queue */
    class SendingThread extends Thread {
	
	final InetAddress sendToAddr;
	final int sendToPort;
	final int reconnSeconds=5, reconnTries=5;

	boolean closeDown = false;
	boolean canSend   = false;

	SendingThread( InetAddress addr, Integer port ) {
	    sendToAddr=addr;
	    sendToPort=(port==null)?(32320):(port);
	}
	
	public void run () {

	    // (re)open the connection
	    canSend = false;
	    System.out.println("SND: (re)connect: "+sendToAddr+" port "+sendToPort);
	    Socket conn = null;
	    OutputStream outStr = null;
	
	    for (int retry=reconnTries; retry>0; retry--) {
		try {
		    conn   = new Socket( sendToAddr, sendToPort );
		    outStr = conn.getOutputStream();
		} catch (IOException e ) {
		    System.err.println("Failed to open "+e);
		    System.err.println(String.format("-> Retrying (%d), in %d sec",
			retry, reconnSeconds));
		}
		if (conn!=null && outStr != null) break;
		
		try {
		    this.sleep( reconnSeconds*1000);
		} catch (InterruptedException e) {
		    if (closeDown) return;
		}
	    }
	  
	    // if connection has failed
	    if (conn==null || outStr==null) {
		System.err.println("Connection has failed: "+sendToAddr+" port "+
		    sendToPort);
		return;
	    }

	    System.out.println("SND: connection open: "+sendToAddr+" port "+sendToPort);
	    canSend = true;

	    // wait for images in the queue, if present send them
	    while (true) {
		// get an image
		ImageWrapper iw;
		try {
		    iw = imageQueue.poll( 2500, TimeUnit.MILLISECONDS ); 
		} catch (InterruptedException e) {
		    iw = null;
		}

		if (iw!=null) {
		    iw.writeData( outStr );		    
		} else {
		    System.out.println("SND: waiting for images, (for "+sendToAddr+")");
		}

		// check if the thread needs to stop
		if (closeDown) {
		    try {
			conn.close();
		    } catch (IOException e) {
			System.err.println("SND: Failed to close connection: "+e);
			return;
		    }
		    System.out.println("SND: closed connection as requested, stopping send thread");
		    return;
		}
	    }
	}

    }
    

    

    /** For testing ... */
    public static void main(String [] arg) 
	throws java.net.UnknownHostException {

	if (arg.length<2) {
	    System.out.println("Usage: delay(ms) host1 [host2] [host3...]");
	    return;
	}

	int delay = Integer.parseInt(arg[0]);
	String [] hosts = new String[arg.length-1];
	System.arraycopy( arg, 1, hosts, 0 , hosts.length);

	System.out.println(" delay: "+delay);

	ImageSender nl = new ImageSender(hosts);

	int count=0;
	short [] tmp = new short[512*512];
	
	while (true) {

	    try {
		Thread.sleep( delay );
	    }
	    catch ( InterruptedException e) {
		System.err.println("ERR: "+e);
		return;
	    }

	    ImageWrapper iw = new ImageWrapper(512,512);

	    for (int y=0; y<512; y++)
	    for (int x=0; x<512; x++)
		tmp[x+y*512] = (short)((x*16 + y%16 + count*32 )%2048);

	    iw.copy(tmp, 512, 512);
	    iw.setPosAB(1,2);
	    iw.setPos012(0,1,count++);

	    boolean ret = nl.queueImage( iw );
	    if (!ret)
		System.out.println("dropped frame");
	}
    }

}

