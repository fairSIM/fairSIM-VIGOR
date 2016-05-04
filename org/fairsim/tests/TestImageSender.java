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

package org.fairsim.tests;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ShortProcessor;

import org.fairsim.transport.ImageSender;
import org.fairsim.transport.ImageWrapper;

import org.fairsim.utils.Tool;

public class TestImageSender {
    
    /** Start from the command line to run the plugin */
    public static void main( String [] arg ) throws java.net.UnknownHostException {

	if (arg.length<5) {
	    System.out.println("TIFF-file white-interval channel_nr delay_us host1 [host2]");
	    return;
	}
	

	// setup connections	
	ImageSender isend = new ImageSender();
	for (int i=4; i<arg.length; i++) {
	    isend.connect( arg[i], null );
	    Tool.trace("Using connection to: "+arg[i]);
	}

	// open the TIFF  stack
	ImagePlus ip = IJ.openImage(arg[0]);
	ImageStack iSt = ip.getImageStack();

	final int stackLen = iSt.getSize();
	int stackPos = 0;

	// parameters
	final int whiteFrame = Integer.parseInt( arg[1] );
	final boolean doWhiteFrame = (whiteFrame>0);
	final int channelNr = Integer.parseInt( arg[2] );
	final int delayus = Integer.parseInt( arg[3] );

	Tool.trace("Image file: "+arg[0]);
	if (doWhiteFrame)
	    Tool.trace("Sync frame all "+whiteFrame+" frames");
	else
	    Tool.trace("no sync frames");

	Tool.trace("delay between frames: "+delayus+" microseconds");

	final int width  = iSt.getWidth();
	final int height = iSt.getHeight();
	
	int count=0;

	Tool.Timer t1 = Tool.getTimer();
	t1.start();

	// loop sending images
	while ( true ) {
	    
	    ImageWrapper iwrap = null;
	   
	    long starttime = System.nanoTime();

	    if ( doWhiteFrame && (count%(whiteFrame+1))==0 ) {
		// insert white frame if required
		short [] white = new short[width*height];
		short [] black = new short[width*height];
		for (int i=0; i<width*height;i++) {
		    white[i] = (short)(10000+Math.random()*500);
		    black[i] = (short)(1000+Math.random()*500);
		}
		
		iwrap = ImageWrapper.copyImage(white,width,height,0,1,0,0,count);
		iwrap.setPos012( 0, channelNr, 0);
		isend.queueImage( iwrap );
		
		iwrap = ImageWrapper.copyImage(black,width,height,0,1,0,0,count);
		iwrap.setPos012( 0, channelNr, 0);
		isend.queueImage( iwrap );
	    } else {
		// add image from stack
		ShortProcessor sp = 
		    iSt.getProcessor(stackPos+1).convertToShortProcessor();
		
		iwrap = ImageWrapper.copyImage( 
		    (short[])sp.getPixels(), width, height, 0,0,0,0, count);

		iwrap.setPos012( 0, channelNr, 0);

		stackPos = (stackPos+1)%((stackLen/whiteFrame)*whiteFrame);
	    
		isend.queueImage( iwrap );
	    }

	    count++;

	   
	    if (count%25==0) {
		t1.stop();
		Tool.trace(String.format("Frames send %6d (%7.2f ms/fr, %7.2f fps)", 
		    count, t1.msElapsed()/25, (25*1000)/t1.msElapsed()));
		t1.start();
	    }
	   
	   // wait the delay time for adding new image
	    if (delayus > 10000) {
		try {
		    Thread.sleep( (delayus-5000)/1000 );
		}
		catch ( InterruptedException e) {
		    System.err.println("ERR: "+e);
		    return;
		}
	    } 
	    while ( System.nanoTime() < starttime+delayus*1000) {};

	}



    }
}
