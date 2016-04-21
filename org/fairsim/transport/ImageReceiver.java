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

import java.io.InputStream;
import java.io.IOException;

import java.net.Socket;
import java.net.ServerSocket;
import java.net.InetAddress;

import java.util.List;
import java.util.ArrayList;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

public class ImageReceiver {

    private final int width, height;
    private final BlockingQueue<ImageWrapper> imageQueue; 
    private final BlockingQueue<ImageWrapper> recycledWrapperQueue; 
    private ConnectionHandler ch = null;
    private List<Notify> listeners = new ArrayList<Notify>(2);

    /** Instance to receive images over the network.
     *  The ImageReceiver forks a server thread (see {@link #startReceiving}),
     *  which fills a queue of incoming images.  The queue has to be drained
     *  (see {@link #queue}), otherwise incoming images will be discarded.
     *
     *	@param queueSize how many images to buffer
     *	@param maxWidth  maximum width of incoming image, larger will be discarded
     *	@param maxHeight maximum height of incoming image, larger will be discarded
     * */
    public ImageReceiver(int bufferSize, int maxWidth, int maxHeight) {
	width=maxWidth; height=maxHeight;
	imageQueue = new ArrayBlockingQueue<ImageWrapper>(bufferSize);
	recycledWrapperQueue = new ArrayBlockingQueue<ImageWrapper>(bufferSize);
    }

    
    /** Forks a server thread to receive images. Incoming connections
     *  will fork further threads, so multiple connections are
     *  possible.
     *
     *  @param bind The address to bind to, maybe 'null' to bind to all
     *  @param port The port to bind to, maybe 'null' for default (32320)
     * 
     * */
    public void startReceiving( InetAddress bind, Integer port ) 
	throws IOException {
	
	ConnectionHandler ch = new ConnectionHandler(bind, port);
	ch.start();
    }
    

    /** This thread only accepts connections, delegates receiving data to
     *  sub-threads */
    private class ConnectionHandler extends Thread {
	
	final InetAddress bindAddr;
	final int bindPort;
	final ServerSocket serv ;

	ConnectionHandler( InetAddress bind, Integer port ) throws IOException {
	    bindAddr=bind;
	    bindPort=(port==null)?(32320):(port);
	    serv  = new ServerSocket( bindPort, 0, bindAddr );
	}
	
	public void run () {
   
	    message("Awaiting connections: "+bindAddr+" prt "+bindPort,false,false);

	    // accept connections
	    while (true) {
		ImageServer iServ = new ImageServer();
		try {
		    iServ.sckt = serv.accept();
		    iServ.inStr = new java.io.BufferedInputStream( 
			iServ.sckt.getInputStream(), 4096*1024);
		} catch (Exception e) {
		    message("Accept error "+e, true, true);
		    return;
		}
		// fork a thread to handle that connection 
		iServ.start();
	    }
	}

    }
    

    /** This thread takes a TCP connection and received the images send on
     *  that connection, terminates as soon as the connection closes */
    class ImageServer extends Thread {

	Socket sckt = null;
	InputStream inStr = null;

	/** Run the server thread */
	public void run () {

	    message("forked receiving thread",false,false);
	    long count=0;
    
	    // read from the input stream
	    while (true) {
	    

		// optimization: try if we can reuse an recycled ImageWrapper
		ImageWrapper recvImage = recycledWrapperQueue.poll();
		if (recvImage == null) recvImage = new ImageWrapper(width,height);
		int r0=0, r1=0;
		
		try {
		    r0 = recvImage.readHeader( inStr );
		    if (r0>0)
			r1 = recvImage.readData( inStr );	
		} catch ( Exception e ) {
		    message("failed to receive image: "+e,true,false);
		    return;
		}
		if (r0<0 || r1<0) {
		    //reopen port
		    message(String.format(
			"conn dropped (after %d img)",count),false,false);
		    break;
		}
		// put the image into the queue
		imageQueue.offer( recvImage );
		count++;
	    }

	}
    
    }

    // ---- Image acess ----

    /** Take the next image, blocking till it is available.
     *	This amounts to calling 'take' on the underlying 'BlockingQueue' */
    public ImageWrapper takeImage() {
	ImageWrapper iw = null;
	try {
	    iw = imageQueue.take();
	} catch (InterruptedException e) {
	    message( "Taking images from queue interrupted",true,false);
	    iw = null;
	}
	return iw;
    }
    

    /** Queues an ImageWrapper for recycling */
    public void recycleWrapper( ImageWrapper iw ) {
	recycledWrapperQueue.offer( iw );
    }	



    // ---- Notification handling ----

    private void message( String m, boolean err, boolean fail) {
	for ( Notify i: listeners)
	    i.message( m, err, fail );
    }

    /** Add a notification listener */
    public void addListener( Notify l ) {
	listeners.add(l);
    }
    /** Remove a notification listener */
    public void removeListener( Notify l) {
	listeners.remove(l);
    }
    /** Server notifications */
    public interface Notify {
	/** Gets called when the ImageReeicer reports an event.
	 *  Please note: Currently, notifications are send from within the
	 *  server threads as blocking, so code should NOT spend too
	 *  much time to react to them.
	 *  @param message A clear-text message what happened
	 *  @param isError If the message is to be considered an error
	 *  @param isFatal If an error occurred that stopped the server
	 * */
	public void message( String message, boolean isError, boolean isFatal);
    }


    /** For testing ... */
    public static void main(String [] arg) 
	throws java.net.UnknownHostException, java.io.IOException {

	ImageReceiver nl = new ImageReceiver(16,512,512);

	
	nl.addListener( new ImageReceiver.Notify() {
	    public void message( String m, boolean err, boolean fail) {
		if (err) {
		    System.err.println(((fail)?("FAIL:"):("err: "))+m);
		} else {
		    System.out.println("srv: "+m);
		}
	    }
	});
	
	nl.startReceiving( null, null);
    }


}

