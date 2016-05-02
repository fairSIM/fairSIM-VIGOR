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

package org.fairsim.livemode;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;

import java.util.Map;
import java.util.TreeMap;

import org.fairsim.transport.ImageReceiver;
import org.fairsim.transport.ImageWrapper;
import org.fairsim.utils.Tool;
import org.fairsim.linalg.MTool;


/** Takes raw images from the network listener, syncs
 * to SIM sequence, passes them to ReconstructionRunner */
public class SimSequenceExtractor {

    final ImageReceiver imgRecv;
    final ReconstructionRunner reconRunner; 
    final int nrChannels;
    final int seqCount;

    final PerChannelBuffer [] channels;
    private Map< Integer, PerChannelBuffer > channelMapping;

    /** Links ir to rr */
    public SimSequenceExtractor( int seqCount, 
	ImageReceiver ir, ReconstructionRunner rr ) {
	this.imgRecv = ir;
	this.reconRunner = rr;
	this.nrChannels = rr.nrChannels;
	this.seqCount = seqCount;

	channelMapping = new TreeMap<Integer, PerChannelBuffer >() ;
	channels = new PerChannelBuffer[ nrChannels ];
	for (int i=0; i<nrChannels; i++) {
	    channels[i] = new PerChannelBuffer(9*9*10);	
	    channelMapping.put( reconRunner.getChannel(i).chNumber, channels[i]);
	}
    
	// start an ImageSorter
	ImageSorter is = new ImageSorter();
	is.start();

	// start the per-channel sequence detection
	for ( PerChannelBuffer pcb : channels )
	    pcb.start();

	JoinedChannelBuffer jcb = new JoinedChannelBuffer();
	jcb.start();
    
	Tool.trace("Image sequency detection started");
    }


    /** Take images for the gereral queue, sort them by channel */
    class ImageSorter extends Thread {
	@Override
	public void run() {
	    
	    while (true) {
		ImageWrapper iw = imgRecv.takeImage();

		int chNr = iw.pos1();	// pos1 holds the data packets image channel
		PerChannelBuffer pc = channelMapping.get(chNr);
		if (pc==null) {
		    Tool.trace("ImgSort: received data packet w/o channel");
		} else {
		    pc.pushImg( iw );
		}
	    }
	}
    }


    /** Takes tuples, triples, etc of SIM images, and
     *  forwards them for reconstruction */
    class JoinedChannelBuffer extends Thread {

	public void run() {

	    while (true) {

		short [][][] res = new short[nrChannels][][];

		try {
		    for (int c=0; c< nrChannels; c++)
			res[c] = channels[c].simSeq.take();
		} catch (InterruptedException e) {
		    Tool.trace("Channel joiner interrupted, why?");
		    continue;
		}

		reconRunner.queueImage( res );
	    }
	}

    }




    /** Sorts through the raw images, waiting for a sync frame, 
     *  assembles SIM sequences */
    class PerChannelBuffer extends Thread {
	
	BlockingQueue<ImageWrapper> rawImgs;
	BlockingQueue<short [][]>     simSeq;
	final int queueSize;
	int missedRaw =0;
	int missedSim =0;
	int noSyncSince=0;

	int queryRaw() { return rawImgs.size(); }
	int querySim() { return simSeq.size(); }

	PerChannelBuffer(int queueSize) {
	    this.queueSize = queueSize;
	    rawImgs = new ArrayBlockingQueue<ImageWrapper>( queueSize );
	    simSeq  = new ArrayBlockingQueue<short [][]>( queueSize );
	}
   
	void pushImg( ImageWrapper iw ) {
	    boolean ok =  rawImgs.offer(iw);
	    if (!ok)
		missedRaw++;
	}

	/** Sequence detection, emptying rawImgs, filling simSeq */
	@Override
	public void run ( ) {

	    final int nrRawPerSeq = reconRunner.nrDirs*reconRunner.nrPhases;

	    while ( true ) {
		try {

		    // detect a bright frame (avr > 10000)
		    ImageWrapper iw = rawImgs.take();
		    short pxl [] = iw.getPixels();
		    
		    if (MTool.avr_ushort( pxl ) < 10000) {
			noSyncSince++;
			continue;
		    }

		    // ignore the next frame
		    rawImgs.take();


		    // copy the next n x m frames
		    for (int k=0; k<seqCount; k++) {
			
			short [][] simPxls = new short[nrRawPerSeq][];
			
			for (int i=0; i<nrRawPerSeq; i++) {
			    iw = rawImgs.take();
			    simPxls[i] = iw.getPixels();
			}
		    
			boolean ok = simSeq.offer( simPxls );
			if (!ok) missedSim++;
		    }
		

		} catch ( InterruptedException e ){
		    Tool.trace("Image sorting thread interupted, why?");
		}
	    
	    }

	}
    }



}
