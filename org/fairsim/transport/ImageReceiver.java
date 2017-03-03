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

import javax.swing.JFrame;

import org.fairsim.utils.Tool;
import org.fairsim.sim_gui.PlainImageDisplay;


public class ImageReceiver {

    private int width, height;
    private final BlockingQueue<ImageWrapper> imageQueue; 
    //private final BlockingQueue<ImageWrapper> recycledWrapperQueue; 
    private ConnectionHandler ch = null;

    private ImageDiskWriter imageWriter = null;

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
	//recycledWrapperQueue = new ArrayBlockingQueue<ImageWrapper>(bufferSize);
    }

    public void setImageSize(int width, int height) {
        this.width = width;
        this.height = height;
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
   
	    Tool.trace(" -net- Awaiting connections: "+bindAddr+":"+bindPort);

	    // accept connections
	    while (true) {
		ImageServer iServ = new ImageServer();
		try {
		    iServ.sckt = serv.accept();
		    iServ.inStr = new java.io.BufferedInputStream( 
			iServ.sckt.getInputStream(), 4096*1024);
		} catch (Exception e) {
		    Tool.error("Accept error "+e, true);
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

	    Tool.trace(" -net- new connection, forked receiving thread");
	    long count=0;
    
	    // read from the input stream
	    while (true) {
	    

		// optimization: try if we can reuse an recycled ImageWrapper
		//ImageWrapper recvImage = recycledWrapperQueue.poll();
		//if (recvImage == null) recvImage = new ImageWrapper(width,height);
		
		ImageWrapper recvImage = new ImageWrapper(width,height);
		int r0=0, r1=0;
		
		try {
		    r0 = recvImage.readHeader( inStr );
		    if (r0>0)
			r1 = recvImage.readData( inStr );	
		} catch ( Exception e ) {
		    Tool.error("failed to receive image: "+e,false);
		    return;
		}
		if (r0<0 || r1<0) {
		    //reopen port
		    Tool.trace(String.format(
			"- net- conn dropped (after %d img)",count));
		    break;
		}
		// put the image into the queue
		imageQueue.offer( recvImage );
		if ( imageWriter != null)
		    imageWriter.saveImage( recvImage );
		
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
	    Tool.error( "Taking images from queue interrupted",false);
	    iw = null;
	}
	return iw;
    }
    

    /* Queues an ImageWrapper for recycling 
    public void recycleWrapper( ImageWrapper iw ) {
	recycledWrapperQueue.offer( iw );
    } */

    /** Incoming images will be passed to the 
     * provided writer for disk storage. Set to 'null'
     * to deactivate */
    public void setDiskWriter( ImageDiskWriter iw ) {
	imageWriter = iw;
    }

    
    /** For testing ... */
    public static void main(String [] arg) 
	throws java.net.UnknownHostException, java.io.IOException {

	if (arg.length<=2) {
	    System.out.println("Usage: width height (no)display");
	    return;
	}

	final int width = Integer.parseInt( arg[0] );
	final int height = Integer.parseInt( arg[1] );

	boolean showDisplay = false;


	PlainImageDisplay displ01 = null;
	if (arg[2].equals("display")) {
	    displ01 = new PlainImageDisplay(width, height);
	    JFrame mainFrame = new JFrame("Test display");
	    mainFrame.add( displ01.getPanel());
	    mainFrame.pack();
	    mainFrame.setVisible(true);	    
	}   


	ImageReceiver nl = new ImageReceiver(400,width,height);
	nl.startReceiving( null, null);
	
	Tool.Timer t1 = Tool.getTimer();
	long count=0,lastcount=0;
    
	while (true) {
	    ImageWrapper iw = nl.takeImage();
	    count++;
	    if (count%100==0) {
		t1.stop();
		Tool.trace(String.format("received %10d images: %7.2f fps",
		    count, 100/(t1.msElapsed()/1000)));
		t1.start(); 
	    }
	   
	    if (count%10==0 && displ01!=null) {
		displ01.newImage(0, iw.getPixels());
		displ01.refresh();
	    }
		
	     
	}

    



    }


}

