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


import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;

import org.fairsim.linalg.*;
import org.fairsim.sim_algorithm.*;

import org.fairsim.network.ImageReceiver;
import org.fairsim.network.ImageWrapper;

import org.fairsim.sim_gui.PlainImageDisplay;

import org.fairsim.utils.Tool;
import org.fairsim.utils.Conf;
import org.fairsim.utils.SimpleMT;
import org.fairsim.utils.ImageDisplay;

import org.fairsim.accel.AccelVectorFactory;

/** Class to run instant SIM reconstruction with fixed parameters. */
public class TestInstantRecon  {

    final float offset = 1100;

    // buffer for received images
    BlockingQueue< Vec2d.Real []> imgsToReconstruct =
	new ArrayBlockingQueue< Vec2d.Real [] >(10);

    // buffer for reconstructed images
    BlockingQueue< Vec2d.Real > finalImages =
	new ArrayBlockingQueue< Vec2d.Real >(10);
    
    BlockingQueue< Vec2d.Real > finalImagesWidefield =
	new ArrayBlockingQueue< Vec2d.Real >(10);

    final SimParam param;
    final int width, height;

    TestInstantRecon( SimParam p, int w, int h ) {
	param=p; width=w; height=h;
    }

    public void startAllThreads() {



	JFrame frame1 = new JFrame("Widefield (live)");
	JFrame frame2 = new JFrame("SIM recon (live)");
        

	ReconstructorThread rt = new ReconstructorThread();
	rt.start();
	NetworkedReconstruction nr = new NetworkedReconstruction();
	nr.start();
	
	ImageDisplayThread id1 = new ImageDisplayThread(true);
	id1.start();
	
	ImageDisplayThread id2 = new ImageDisplayThread(false);
	id2.start();
	
	JScrollPane pane1 = new JScrollPane( id1.dspl.getPanel(),
	    JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
	    JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
	JScrollPane pane2 = new JScrollPane( id2.dspl.getPanel(),
	    JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED, 
	    JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);
	
	frame1.setContentPane(pane1);
	frame2.setContentPane(pane2);
	//pane1.add( id1.dspl.getPanel());
	//pane2.add( id2.dspl.getPanel());
	frame1.pack();
	frame2.pack();
	frame1.setVisible(true);
	frame2.setVisible(true);


    }

   
   
    /** The actual reconstruction. This pulls
     * raw images from the 'imgsToReconstruct' queue, reconstructs them,
     * puts them into 'finalImages' */
    private class ReconstructorThread extends Thread {
	public void run() {
	    Tool.trace("Setting up reconstruction");

	    // parameters
	    param.setPxlSize( width, 0.08 );
	    final OtfProvider otfPr  = param.otf(); 
	    
	    double apoStr  = 0.99;
	    double apoFWHM = 1.5;
	    double wienParam = 0.05;
    
	    // Setup OTFs, Wiener filter, APO
	    Vec2d.Cplx[] otfV    = Vec2d.createArrayCplx( param.nrBand(), 
				    param.vectorWidth(), param.vectorHeight() );
	    
	    for (int i=0; i<param.nrBand(); i++) {
		otfPr.writeOtfWithAttVector( otfV[i], i, 0,0 );
		otfV[i].makeCoherent();

		Tool.trace(String.format("OTF norm2: %7.4e ",otfV[i].norm2()));
	    }

	    WienerFilter wFilter = new WienerFilter( param );
	    Vec2d.Real wienerDenom = wFilter.getDenominator( wienParam );
	    wienerDenom.makeCoherent();

	    Vec2d.Cplx apoVector = Vec2d.createCplx(2*width,2*height);
	    otfPr.writeApoVector( apoVector, apoStr, apoFWHM);
	    apoVector.makeCoherent();

	    // vectors to store the result
	    Vec2d.Cplx fullResult   = Vec2d.createCplx( param, 2);

	    Vec2d.Cplx [][] inFFT   = new Vec2d.Cplx[param.nrDir()][];
	    for (int a=0; a<param.nrDir(); a++)
		inFFT[a] =  Vec2d.createArrayCplx( param.dir(a).nrPha(), width, height);
	    Vec2d.Cplx [] separate  = Vec2d.createArrayCplx( param.nrBand()*2-1, width, height);
	    Vec2d.Cplx [] shifted   = Vec2d.createArrayCplx( param.nrBand()*2-1, 2*width, 2*height);

	    Tool.Timer tAll = Tool.getTimer();
	    int reconCount=0;

	    // run the reconstruction
	    while ( true ) {

		// retrieve image, run fft
		Vec2d.Real [] imgs = null ;
		try {
		    imgs = imgsToReconstruct.take();
		} catch (InterruptedException e) {
		    Tool.trace("Thread interrupted, frame missed");
		    continue;
		}
	
		tAll.start();
		int count=0;
		Vec2d.Real widefield = Vec.getBasicVectorFactory().createReal2D(width,height);

		for (int a=0; a<param.nrDir(); a++)
		    for (int p=0; p<param.dir(a).nrPha(); p++) {
			
		    Vec2d.Real inImg = imgs[count++];
		    float [] dat = inImg.vectorData();
		    for (int i=0; i<dat.length; i++)
			dat[i] -= offset;
		    //inImg.syncBuffer();

		    SimUtils.fadeBorderCos( inImg , 10);
		    widefield.add( inImg );
		    
		    inFFT[a][p].copy( inImg );
		    inFFT[a][p].fft2d(false);
		}

		finalImagesWidefield.offer( widefield);

		// loop pattern directions
		fullResult.zero();
		for (int angIdx = 0; angIdx < param.nrDir(); angIdx ++ ) {
		    final SimParam.Dir par = param.dir(angIdx);

		    // ----- Band separation & OTF multiplication -------

		    BandSeparation.separateBands( inFFT[angIdx] , separate , 
			    par.getPhases(), par.nrBand(), par.getModulations());

		    for (int i=0; i<(par.nrBand()*2-1) ;i++)  
			separate[i].timesConj( otfV[ (i+1)/2 ]);
		    

		    // ------- Shifts to correct position ----------

		    // band 0 is DC, so does not need shifting, only a bigger vector
		    SimUtils.placeFreq( separate[0],  shifted[0]);
		    
		    // higher bands need shifting
		    for ( int b=1; b<par.nrBand(); b++) {
			int pos = b*2, neg = (b*2)-1;	// pos/neg contr. to band
			SimUtils.placeFreq( separate[pos] , shifted[pos]);
			SimUtils.placeFreq( separate[neg] , shifted[neg]);
		    }

		    for ( int b=1; b<par.nrBand(); b++) {
			int pos = b*2, neg = (b*2)-1;	// pos/neg contr. to band
			SimUtils.fourierShift( shifted[pos] ,  par.px(b),  par.py(b) );
			SimUtils.fourierShift( shifted[neg] , -par.px(b), -par.py(b) );
		    }
		   
		    
		    // sum up to full result 
		    for (int i=0;i<par.nrBand()*2-1;i++)  
			fullResult.add( shifted[i] ); 
		
		} // end direction loop

		// apply wiener filter and APO
		fullResult.times(wienerDenom);
		fullResult.timesConj(apoVector);
		
		fullResult.fft2d(true);

		Vec2d.Real res = 
		    Vec.getBasicVectorFactory().createReal2D(2*width,2*height);

		res.copy(fullResult);
	   
		finalImages.offer(res);
		tAll.hold();
		
		// some feedback
		reconCount++;
		if (reconCount%10==0) {
		    Tool.trace(String.format(
			"reconst:  #%5d %7.2f ms/fr %7.2f ms/raw %7.2f fps(hr) %7.2f fps(raw)", 
			reconCount, tAll.msElapsed()/10, tAll.msElapsed()/(10*param.getImgPerZ()),
			1000./(tAll.msElapsed()/10.), 
			1000./(tAll.msElapsed()/(10.*param.getImgPerZ()))));
		    tAll.stop();
		}

	    }



	}
    }

    /** Received images, waits for sync frames, queues them.
     *  All images received go into the 'imgsToReconstruct' queue. */
    private class NetworkedReconstruction extends Thread {
    
	public void run ( ) {

	    // Setup network link
	    ImageReceiver ir = new ImageReceiver(50,512,512);
	    boolean keepRunning = true;

	    ir.addListener( new ImageReceiver.Notify() {
		public void message( String m , boolean err, boolean fail) {
		    String e = (err)?( (fail)?("FAIL: "):("err: ") ):("net: ");
		    Tool.trace( e + m );
		}
	    });

	    try {
		ir.startReceiving(null,null);	
	    } catch  ( java.io.IOException e ) {
		Tool.trace("Net setup failed: "+e);
		return;
	    }

	    int rawImgCount = param.getImgPerZ();

	    // since all this is on the net: Basic (non-GPU) vectors
	    final VectorFactory bvf = Vec.getBasicVectorFactory();


	    // Big reconstruct loop
	    int count=0;
	    Tool.Timer t1 = Tool.getTimer();
	    while ( keepRunning ) {
	
		Vec2d.Real tmpImgReal   = bvf.createReal2D(width,height);

		// detect white
		while ( true  ) {
		    ir.takeImage().writeToVector( tmpImgReal );
		    double val = tmpImgReal.sumElements() / tmpImgReal.vectorSize();
		    if (val>10000) break;
		} 

		// copy the images following the sync
		Vec2d.Real [] imgs = new Vec2d.Real[rawImgCount] ;
		for (int i=0; i<rawImgCount; i++) {
		    imgs[i] = bvf.createReal2D(width,height);
		    ir.takeImage().writeToVector(imgs[i]);
		}

		// put image into queue
		imgsToReconstruct.offer( imgs );
		if (count%(10)==0) {
		    t1.stop();
		    Tool.trace(String.format(
			"receive:  #%5d %7.2f ms/fr %7.2f ms/raw %7.2f fps(hr) %7.2f fps(raw)", 
			count, t1.msElapsed()/10, t1.msElapsed()/10/rawImgCount, 
			10000./t1.msElapsed(), (10000.*rawImgCount)/t1.msElapsed()));
		    t1.start();
		}
		count++;
	    }

	}
    }

    /** Thread that displays final images */
    private class ImageDisplayThread extends Thread {

	final PlainImageDisplay dspl ;
	final int w, h;
	final boolean widefield;
    
	ImageDisplayThread( boolean wf ) {
	    
	    widefield = wf;
	    w =  width*((widefield)?(1):(2));
	    h = height*((widefield)?(1):(2));
	    dspl = new PlainImageDisplay(w,h);

	}

	public void run() {

	    while (true) {
	
		// get the image
		Vec2d.Real img = null;
		try {
		    if (widefield) 
			img = finalImagesWidefield.take();
		    else
			img = finalImages.take();
		} catch ( InterruptedException e ) {
		    Tool.trace("Display thread interrupted, frame lost");
		    continue;
		}

		// scale the image
		float max = Float.MIN_VALUE;
		float [] dat = img.vectorData();
		for (int i=0; i<w*h; i++)
		    max = Math.max(dat[i],max);
		
		img.scal( 4096/max );

		// set image
		dspl.newImage( img );

	    }
	}


    }




    /** Start from the command line to run the plugin */
    public static void main( String [] arg ) {

	// output usage
	if (arg.length<2) {
	    System.err.println("[JAVA|CUDA] (sim-param.xml)");
	    return;
	}

	// set the accelerator module
	boolean set=false;
	String wd = System.getProperty("user.dir")+"/accel/";
	Tool.trace("loading library from: "+wd);

	VectorFactory avf = null;

	if (arg[0].equals("CUDA")) {
	    System.load(wd+"libcudaimpl.so");
	    Tool.trace("Running with CUDA support now");
	    Vec.setVectorFactory( AccelVectorFactory.getFactory()); 
	    set=true;
	    SimpleMT.useParallel(false);
	}
	if (arg[0].equals("JAVA")) {
	    set=true;
	}
	if (set==false) {
	    System.err.println("pass either: JAVA, CUDA");
	    return;
	}

	// load the parameter set
	SimParam sp = null;
	try {
	    Conf cfg = Conf.loadFile( arg[1] );
	    sp = SimParam.loadConfig( cfg.r() );
	    OtfProvider otf = OtfProvider.loadFromConfig( cfg );
	    sp.otf( otf );
	    if ( otf == null )
		throw new Exception("No OTF found");

	} catch ( Exception e ) {
	    System.err.println("Failed to load parameters: "+e);
	    return;
	}
	
	// start the reconstruction loop
	TestInstantRecon tir = new TestInstantRecon(sp,512,512);
	tir.startAllThreads();
    
    }




}


