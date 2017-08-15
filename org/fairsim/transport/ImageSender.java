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

import java.io.OutputStream;
import java.io.IOException;

import java.net.Socket;
import java.net.InetAddress;

import java.util.List;
import java.util.ArrayList;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

public class ImageSender {


    // buffer for images to send
    public BlockingQueue<ImageWrapper> imageQueue = new ArrayBlockingQueue<ImageWrapper>(2048);
   
    // list of TCP connections to use
    List<SendingThread> connection = new ArrayList<SendingThread>();

    /** Start ImageSender without initial connection */
    public ImageSender() {

    }

    /** Start ImageSender without initial connection, but with given buffer size*/
    public ImageSender(int bufSize) {
	imageQueue = new ArrayBlockingQueue<ImageWrapper>(bufSize);
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

    public void shutdownThreads() {
        for (SendingThread s : connection) {
            s.closeDown = true;
            s.interrupt();
        }
    }

    /** Non-blocking image send. Returns if the image was sucessfully
     * queued for sending. */
    public boolean queueImage( ImageWrapper img ) {
	if (img==null) return false;
	return imageQueue.offer( img);
    }

    public void clearBuffer() {
        imageQueue.clear();
    }
    
    public boolean canSend() {
        for (SendingThread s : connection) {
            if (s.canSend) return s.canSend;
        }
        return false;
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
	    while (!closeDown) {
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
	    System.out.println("Usage: delay(us) host1 [host2] [host3...]");
	    return;
	}

	int delay = Integer.parseInt(arg[0]);
	String [] hosts = new String[arg.length-1];
	System.arraycopy( arg, 1, hosts, 0 , hosts.length);

	System.out.println(" running with delay (us): "+delay);

	ImageSender nl = new ImageSender(hosts);

	short [][] tmp = new short[512][512*512];
	
	// pre-calculate some images
	for (int i=0; i<tmp.length; i++)
	    for (int y=0; y<512; y++)
		for (int x=0; x<512; x++)
		    tmp[i][x+y*512] = (short)((x*16 + y%16 + i*32 )%2048);

	long refTime = System.nanoTime();	

	int count=0;
	while (true) {

	    // TODO: move all this timing stuff into utils.Tool
	    long runAt   = refTime+delay*1000;	// when to run next, in ns
	    long delay_ms = (runAt - System.nanoTime()) / 1000000;

	    // output some warning if the system is too slow
	    if (runAt<0) {
		System.err.println("Delay too slow, system not fast enough to send");
	    }   

	    // only use seep if there are at least 1 ms to sleep
	    if (delay_ms>=10)
	    try {
		Thread.sleep( delay_ms-5 );
	    }
	    catch ( InterruptedException e) {
		System.err.println("ERR: "+e);
		return;
	    }
	    
	    // busy-wait for the rest
	    while ( System.nanoTime() < runAt ) {
	    
	    }   
 
	    // we want 'non-blocking', i.e. set the time BEFORE we use some to send the image
	    refTime=System.nanoTime();

	    ImageWrapper iw = new ImageWrapper(512,512);

	    iw.copy(tmp[count%tmp.length], 512, 512);
	    iw.setPosAB(1,2);
	    iw.setPos012(0,1,count++);

	    boolean ret = nl.queueImage( iw );
	    if (!ret)
		System.out.println("dropped frame");
	
	}
    }

}

