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

import org.fairsim.sim_algorithm.*;
import org.fairsim.accel.*;
import org.fairsim.linalg.*;
import org.fairsim.utils.Tool;

import org.fairsim.utils.Conf;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;

/** Manages a collection of reconstruction threads
 * and parameter fitting. */
public class ReconstructionRunner {
    
    public final int width, height, nrChannels;
    public final int nrDirs, nrPhases, nrBands;
    public final int nrThreads;

    private BlockingQueue<short [][][]> imgsToReconstruct;
    private int missedDueToQueueFull;
    final private PerChannel [] channels;

    BlockingQueue<Vec2d.Real []> finalWidefield;
    BlockingQueue<Vec2d.Real []> finalRecon;
    
    private final ReconstructionThread [] reconThreads ;

    /** Parameters that have to be set per-channel */
    public class PerChannel {
	int chNumber;
	SimParam param;
	float offset;
	String label;
	double wienParam = 0.05; 
	double attStr  = 0.99; 
	double attFWHM = 1.25; 
    }

    public PerChannel getChannel(int i) {
	return channels[i];
    }

    
    /** Queue image set for reconstruction */
    public boolean queueImage( short [][][] imgs ) {
	boolean ok = imgsToReconstruct.offer(imgs);
	if (!ok) missedDueToQueueFull++;
	return ok;
    }

    public ReconstructionRunner( Conf.Folder cfg, 
	VectorFactory avf, String [] whichChannels)  
	throws Conf.EntryNotFoundException {
	this(cfg,avf,whichChannels, true);
    }

    /** Reads from cfg-folder all channels in 'channels' */
    public ReconstructionRunner( Conf.Folder cfg, 
	VectorFactory avf, String [] whichChannels, boolean autostart ) 
	throws Conf.EntryNotFoundException {

	nrThreads = cfg.getInt("ReconThreads").val();
	imgsToReconstruct = new ArrayBlockingQueue<short [][][]>(maxInReconQueue());
	
	finalWidefield = new ArrayBlockingQueue<Vec2d.Real []>(maxInWidefieldQueue());
	finalRecon     = new ArrayBlockingQueue<Vec2d.Real []>(maxInFinalQueue());

	int size  = cfg.getInt("RawPxlCount").val();
	height = width = size;
	double microns = cfg.getDbl("RawPxlSize").val();

	nrPhases = cfg.getInt("NrPhases").val();
	nrDirs   = cfg.getInt("NrAngles").val();
	nrBands  = cfg.getInt("NrBands").val();

	// init per-channel information
	this.nrChannels = whichChannels.length;
	channels = new PerChannel[nrChannels];
	for (int c=0; c<nrChannels; c++)
	    channels[c] = new PerChannel();

	// load initial SIM-param from file
	for (int i=0; i<nrChannels; i++) {
	    // TODO: this should be moved to a "PerChannel" constructor
	    Conf.Folder fld = cfg.cd("channel-"+whichChannels[i]);
	    channels[i].param = SimParam.loadConfig( fld );
	    channels[i].param.setPxlSize( size, microns );
	    OtfProvider otf = OtfProvider.loadFromConfig( fld );
	    channels[i].param.otf( otf );
	    channels[i].offset = (float)fld.getDbl("offset").val();
	    channels[i].chNumber = fld.getInt("ChannelNumber").val();
	    channels[i].label = whichChannels[i];
	}
	

	// create and start reconstruction threads
	reconThreads = new ReconstructionThread[ nrThreads ];
	
	for ( int i=0; i<nrThreads; i++) {
	    reconThreads[i] = new ReconstructionThread( avf ); 
	    if (autostart) {
		reconThreads[i].start();
		Tool.trace("Started recon thread: "+i);
	    }
	}
	// precompute filters for all threads
	setupFilters();

    }


    // ---- MGMT ----

    /** Query how many images are queued for reconstruction */
    public int nrInReconQueue() { return imgsToReconstruct.size(); }
    public int maxInReconQueue() { return nrThreads*16; }
    
    public int nrInWidefieldQueue() { return finalWidefield.size(); }
    public int maxInWidefieldQueue() { return nrThreads*16; }
    
    public int nrInFinalQueue() { return finalRecon.size(); }
    public int maxInFinalQueue() { return nrThreads*16; }
    
    /** Query and reset how many images where missed */
    public int nrMissedImages() {
	int ret = missedDueToQueueFull;
	missedDueToQueueFull=0;
	return ret;
    }




    // ... setup filters ....
    

    public void setupFilters() {

	Tool.Timer t1=Tool.getTimer();
	Tool.Timer t2=Tool.getTimer();

	// first, calculate filters on all CPUs
	Vec2d.Real dampBorder = Vec2d.createReal( width, height);
	dampBorder.addConst(1.f);
	SimUtils.fadeBorderCos( dampBorder , 10);
	    
	Vec2d.Cplx [] otfV = Vec2d.createArrayCplx( nrChannels,   width, height);
	Vec2d.Real [] wien = new Vec2d.Real[ nrChannels ];
	Vec2d.Real [] apo = Vec2d.createArrayReal(  nrChannels, 2*width, 2*height);
	Vec2d.Cplx tmp = Vec2d.createCplx(  2*width, 2*height);

	for (int c=0; c<nrChannels; c++) {
	    getChannel(c).param.otf().setAttenuation(
		getChannel(c).attStr, getChannel(c).attFWHM);
	    getChannel(c).param.otf().writeOtfWithAttVector( otfV[c], 0, 0, 0 );
	    WienerFilter wFilter = new WienerFilter( getChannel(c).param );
	    wien[c] = wFilter.getDenominator( getChannel(c).wienParam );
	    getChannel(c).param.otf().writeApoVector( tmp, 1, 2 );
	    apo[c].copy( tmp );
	}

	t1.stop();
	t2.start();

	// then, copy to every GPU thread   // TODO: concurrency sync here?
	for ( ReconstructionThread r : reconThreads ) {
	    r.dampBorder.copy( dampBorder );
	    r.dampBorder.makeCoherent();
	    for (int c=0; c<nrChannels; c++) {
		r.otfVector[c].copy( otfV[c] );
		r.wienDenom[c].copy( wien[c] );
		r.apoVector[c].copy( apo[c] );
		r.otfVector[c].makeCoherent();
		r.wienDenom[c].makeCoherent();
		r.apoVector[c].makeCoherent();
	    }

	}
	
	t2.stop();

	Tool.trace("Updates filters: "+t1+" "+t2);


    }

    
    /** A single reconstruction thread, pulling raw images
     *  from the queue and reconstructing them to a final image */
    private class ReconstructionThread extends Thread {

	final VectorFactory avf;

	// filter vectors (generated thread-local, so they live on the correct GPU)
	final Vec2d.Cplx [] otfVector;
	final Vec2d.Real dampBorder ;
	final Vec2d.Real [] apoVector, wienDenom;
	    
	Vec2d.Cplx [] fullResult;
	Vec2d.Cplx [] widefield;
	
	Vec2d.Cplx [][][] inFFT ;
	Vec2d.Cplx [][][] separate; 
	Vec2d.Cplx [][][] shifted;

	int maxRecon=0;
	
	final int band2 = nrBands*2-1;

	/** pre-allocate all the vectors */
	ReconstructionThread( VectorFactory v ) {
	    avf = v;
	    otfVector  = avf.createArrayCplx2D(nrChannels, width, height );
	    dampBorder = avf.createReal2D( width, height );
	    apoVector  = avf.createArrayReal2D(nrChannels, 2*width, 2*height );
	    wienDenom  = avf.createArrayReal2D(nrChannels, 2*width, 2*height );
	    
	    fullResult =avf.createArrayCplx2D( nrChannels, 2*width, 2*height );
	    widefield = avf.createArrayCplx2D( nrChannels, width, height );
	    inFFT = avf.createArrayCplx2D( 
		nrChannels, nrDirs, nrPhases, width, height );
	    separate  = avf.createArrayCplx2D(
		nrChannels, nrDirs, band2, width, height);
	    shifted   = avf.createArrayCplx2D(
		nrChannels, nrDirs, band2, 2*width, 2*height);
	}

	public void run() {
	   
	    // vectors for intermediate results

	    Tool.Timer tAll = Tool.getTimer();
	    int reconCount=0;

	    // run the reconstruction loop
	    while ( true ) {

		// retrieve images from queue
		short [][][] imgs = null;
		try {
		    imgs = imgsToReconstruct.take();
		} catch (InterruptedException e) {
		    Tool.trace("Thread interrupted, frame missed");
		    continue;
		}

		
		tAll.start();

		// zero the collecting vectors
		for (int c=0; c<nrChannels; c++) {
		    widefield[c].zero();
		    fullResult[c].zero();
		}
		
		// generate result vectors cache on the CPU
		Vec2d.Real [] cpuWidefield =
		    Vec.getBasicVectorFactory().createArrayReal2D(
			nrChannels, width,height);
		
		Vec2d.Real [] cpuRes = 
		    Vec.getBasicVectorFactory().createArrayReal2D(
		    nrChannels, 2*width,2*height);

		// run all input through fft
		for (int c=0; c<nrChannels; c++) {
		    
		    int count=0;
		    
		    for (int a=0; a<nrDirs; a++)
			for (int p=0; p<nrPhases; p++) {
			    
			short [] inImg = imgs[c][count++];
			inFFT[c][a][p].setFrom16bitPixels( inImg );

			// fade borders
			inFFT[c][a][p].times( dampBorder );
			
			// TODO: this would be a place to add the compensation 

			// add them up to widefield
			widefield[c].add( inFFT[c][a][p] );
			inFFT[c][a][p].fft2d(false);
		    }
		
		    // copy back wide-field
		    widefield[c].scal(1.f/(nrDirs*nrPhases));
		    cpuWidefield[c].copy( widefield[c] );
		}

		finalWidefield.offer( cpuWidefield );

		
		// loop channel
		for (int channel=0; channel<nrChannels; channel++) {
		    
		    // loop pattern directions
		    for (int angIdx = 0; angIdx < nrDirs; angIdx ++ ) {
			final SimParam.Dir par = getChannel(channel).param.dir(angIdx);

			// ----- Band separation & OTF multiplication -------
			BandSeparation.separateBands( inFFT[channel][angIdx], separate[channel][angIdx], 
				par.getPhases(), par.nrBand(), par.getModulations());
			
			for (int i=0; i<band2 ;i++)  
			    separate[channel][angIdx][i].timesConj( otfVector[channel] );

			// ------- Shifts to correct position ----------

			// band 0 is DC, so does not need shifting, only a bigger vector
			SimUtils.placeFreq( separate[channel][angIdx][0], shifted[channel][angIdx][0]);
			
			// higher bands need shifting
			for ( int b=1; b<par.nrBand(); b++) {
			    int pos = b*2, neg = (b*2)-1;	// pos/neg contr. to band
			    
			    SimUtils.pasteAndFourierShift( 
				    separate[channel][angIdx][pos], shifted[channel][angIdx][pos],
				    par.px(b),  par.py(b), true );
			    SimUtils.pasteAndFourierShift( 
				    separate[channel][angIdx][neg], shifted[channel][angIdx][neg], 
				    -par.px(b), -par.py(b), true );
			}
		       
			// sum up to full result 
			for (int b=0;b<band2;b++)  
			    fullResult[channel].add( shifted[channel][angIdx][b] ); 

		    
		    } // end direction loop

		    // apply wiener filter and APO
		    fullResult[channel].times(wienDenom[channel]);
		    fullResult[channel].times(apoVector[channel]);
		    
		    fullResult[channel].fft2d(true);
		    cpuRes[channel].copy( fullResult[channel] );
		} // end per-channel loop
	
		finalRecon.offer( cpuRes );

		//finalImages.offer(res);
		tAll.hold();
		
		// some feedback
		reconCount++;

		if (maxRecon>0 && reconCount>=maxRecon)
		    break;

		if (reconCount%10==0) {
		    int rawImgs = nrChannels*nrDirs*nrPhases;
		    Tool.trace(String.format(
			"reconst:  #%5d %7.2f ms/fr %7.2f ms/raw %7.2f fps(hr) %7.2f fps(raw)", 
			reconCount, tAll.msElapsed()/10, tAll.msElapsed()/(10*rawImgs),
			1000./(tAll.msElapsed()/10.), 
			1000./(tAll.msElapsed()/(10.*rawImgs))));
		    tAll.stop();
		}

	    }
	}

    }


    /** To run the ReconstructionThreads through the NVidia profiler */
    public static void main( String [] args ) throws Exception {
	
	if (args.length<3) {
	    System.out.println("usage: config.xml nrThread nrImages");
	    return;
	}

	
	String wd = System.getProperty("user.dir")+"/accel/";
	System.load(wd+"libcudaimpl.so");
	VectorFactory avf = AccelVectorFactory.getFactory();
	Conf cfg = Conf.loadFile( args[0] );

	ReconstructionRunner rr = new ReconstructionRunner(
	    cfg.r().cd("vigor-settings"), avf, new String [] {"568"}, false);

	// warm-up fft
	avf.createCplx2D(512,512).fft2d(false);
	avf.createCplx2D(1024,1024).fft2d(false);

	int nrThreads = Integer.parseInt(args[1]);
	int nrCount   = Integer.parseInt(args[2]);
	
	for (int i=0; i<nrThreads*nrCount*4;i++)	
	    rr.imgsToReconstruct.offer( new short[1][15][512*512] );


	Tool.Timer t1 = Tool.getTimer();

	AccelVectorFactory.startProfiler();

	// start the n threads
	for (int i=0; i<nrThreads; i++) {
	    rr.reconThreads[i].maxRecon = nrCount;
	    rr.reconThreads[i].start();

	}

	// join the n threads
	for (int i=0; i<nrThreads; i++)
	    rr.reconThreads[i].join();

	AccelVectorFactory.stopProfiler();

	t1.stop();

	int nrFrames = nrCount*nrThreads;
	Tool.trace("Timing "+t1+" for "+nrFrames+": "+
	    String.format(" %7.4f fps ", nrFrames*1000/t1.msElapsed()));

	System.exit(0);
    }





}

/** Parameter fitter, on CPU */
/*
// see if we should rerun a parameter fit
		    if ( runRefit ) {
			final int lb = 1;
			final int hb = (par.nrBand()==3)?(3):(1);
			final int nBand = par.nrBand()-1;

			double [] peak = 
			    Correlation.fitPeak( separate[0], separate[hb], 0, 1, 
				otfPr, -par.px(nBand), -par.py(nBand), 
				0.05, 2.5, null );
		
			Cplx.Double p1 = 
			    Correlation.getPeak( separate[0], separate[lb], 
				0, 1, otfPr, peak[0]/nBand, peak[1]/nBand,0.05 );

			Cplx.Double p2 = 
			    Correlation.getPeak( separate[0], separate[lb], 
				0, 1, otfPr, peak[0], peak[1], 0.05 );

			Tool.trace(
			    String.format("Peak: (dir %1d): fitted -->"+
				" x %7.3f y %7.3f p %7.3f (m %7.3f)", 
				angIdx, peak[0], peak[1], p1.phase(), p1.hypot() ));
	
			par.setPxPy( -peak[0], -peak[1] );
			par.setPhaOff( p1.phase() );
			par.setModulation( 1, p1.hypot() );
			par.setModulation( 2, p2.hypot() );

		    }
*/





