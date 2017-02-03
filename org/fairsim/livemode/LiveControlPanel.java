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

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JButton;
import javax.swing.BoxLayout;
import javax.swing.BorderFactory;
import javax.swing.JProgressBar;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.text.DefaultCaret;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;

import java.awt.Color;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.GridLayout;

import java.util.Arrays;

import org.fairsim.utils.Conf;
import org.fairsim.sim_algorithm.SimParam;
import org.fairsim.sim_algorithm.OtfProvider;
import org.fairsim.transport.ImageReceiver;
import org.fairsim.transport.ImageDiskWriter;
import org.fairsim.transport.ImageWrapper;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.accel.AccelVectorFactory;
import org.fairsim.controller.ClientGui;
import org.fairsim.registration.Registration;
import org.fairsim.linalg.Vec;
import org.fairsim.sim_gui.PlainImageDisplay;

import org.fairsim.linalg.Vec2d;
import org.fairsim.controller.ClientGui;
import org.fairsim.utils.Tool;
import org.fairsim.utils.SimpleMT;

/** Provides the control interface for live mode */
public class LiveControlPanel {

    boolean isRecording = false;    // if the raw stream is recorded

    final JProgressBar networkBufferBar;
    final JProgressBar reconBufferInputBar;
    final JProgressBar reconBufferOutputBar;
    
    final JProgressBar fileBufferBar;

    // The different threads in use:
    final ImageDiskWriter	liveStreamWriter;	
    final ImageReceiver		imageReceiver;
    final ReconstructionRunner  reconRunner;
    final SimSequenceExtractor  seqDetection;
    
    
    final PlainImageDisplay	wfDisplay;
    final PlainImageDisplay	reconDisplay;

    final JTextArea  statusField;
    final JTextField statusMessage;

    JButton [] syncButtons ;

    public LiveControlPanel(final Conf.Folder cfg, 
	VectorFactory avf, String [] channels) 
	throws Conf.EntryNotFoundException, java.io.IOException {
	
	// get parameters
	final int imgSize = cfg.getInt("NetworkBuffer").val();

	//  ------- 
	//  initialize the GUI
	//  ------- 
	
	JFrame mainFrame = new JFrame("Live SIM control");
	JPanel mainPanel = new JPanel();
	mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));

	// GUI - buffering
	JPanel reconBuffersPanel = new JPanel();
	reconBuffersPanel.setBorder(BorderFactory.createTitledBorder(
	    "Reconstruction buffers") );
	reconBuffersPanel.setLayout( new GridLayout( 3, 1,2,2 ));

	networkBufferBar = new JProgressBar();
	networkBufferBar.setString("network input buffer");
	networkBufferBar.setStringPainted(true);

	reconBufferInputBar = new JProgressBar();
	reconBufferInputBar.setString("recon input buffer");
	reconBufferInputBar.setStringPainted(true);
	
	reconBufferOutputBar = new JProgressBar();
	reconBufferOutputBar.setString("recon output buffer");
	reconBufferOutputBar.setStringPainted(true);
	
	reconBuffersPanel.add( networkBufferBar );
	reconBuffersPanel.add( reconBufferInputBar );
	reconBuffersPanel.add( reconBufferOutputBar );

	mainPanel.add( reconBuffersPanel );

	// GUI - image record function
	JPanel recorderPanel = new JPanel();
	recorderPanel.setBorder(BorderFactory.createTitledBorder(
	    "raw Stream recording") );
	recorderPanel.setLayout( new GridLayout( 3, 1,2,2 ));
	
	final JTextField filePrefix = new JTextField("VIGOR", 30);

	final JButton recordButton = new JButton("record");
	recordButton.addActionListener( new ActionListener() {
	    public void actionPerformed( ActionEvent e ) {
		if (!isRecording) {
		    recordButton.setForeground(Color.RED);
		    try {
			liveStreamWriter.startRecording( filePrefix.getText());
		    } catch (Exception ex) {
			throw new RuntimeException(ex);
		    }
		    isRecording = true;
		}
		else {
		    recordButton.setForeground(Color.BLACK);
		    liveStreamWriter.stopRecording();
		    isRecording = false;
		}
	    };
	});

	final JButton bufferClearButton = new JButton("buffer clear / resync");
	bufferClearButton.addActionListener( new ActionListener() {
	    public void actionPerformed( ActionEvent e ) {
		seqDetection.clearBuffers();	
	    };
	});


	fileBufferBar = new JProgressBar();
	fileBufferBar.setString("save buffer");
	fileBufferBar.setStringPainted(true);

	recorderPanel.add( recordButton );
	recorderPanel.add( bufferClearButton );
	recorderPanel.add(filePrefix);
	recorderPanel.add(fileBufferBar);
	mainPanel.add(recorderPanel);
        
        //mainPanel.add(new RegistrationPanel(avf, cfg, channels));

	JButton fitPeakButton = new JButton("run parameter fit");
	fitPeakButton.addActionListener( new ActionListener() {
	    public void actionPerformed( ActionEvent e ) {
		//rt.triggerParamRefit();
	    };
	});

	// error output and such
	JPanel statusPanel = new JPanel();
	statusPanel.setBorder(BorderFactory.createTitledBorder(
	    "status messages") );
	statusField = new JTextArea(30,60);
	statusField.setEditable(false);
	DefaultCaret cr = (DefaultCaret)statusField.getCaret();
	cr.setUpdatePolicy( DefaultCaret.ALWAYS_UPDATE );
	JScrollPane statusScroller = new JScrollPane( statusField );
	statusPanel.add( statusScroller );
	mainPanel.add(statusPanel);
	
	final int size = cfg.getInt("RawPxlCount").val();
	final int nrCh = channels.length;

	JPanel statusPanel2 = new JPanel();
	syncButtons = new JButton[nrCh];

	for ( int c=0; c<nrCh; c++) {
	    syncButtons[c] = new JButton("");
	    syncButtons[c].setEnabled(false);
	    statusPanel2.add( syncButtons[c] );
	}

	statusMessage = new JTextField(40);
	statusMessage.setEditable(false);

	statusPanel2.add( statusMessage );

	mainPanel.add(statusPanel2);

	// redirect log output
	Tool.setLogger( new Tool.Logger() {
	    public void writeTrace( String e ) {
		statusField.append(e+"\n");
	    }
	    public void writeShortMessage( String e) {
		statusMessage.setText(e);
	    }
	    public void writeError( String e, boolean fatal) {
		statusField.append( (fatal)?("FATAL err: "):("ERR :: ")+e+"\n");
	    }
	});
    
        
	
	//  ------- 
	//  initialize the components
	//  ------- 


	// network receiver and image storage
	int netBufferSize   = cfg.getInt("NetworkBuffer").val();
	imageReceiver	    = new ImageReceiver(netBufferSize,size,size);
	
	String saveFolder   = cfg.getStr( "DiskFolder" ).val();
	int diskBufferSize  = cfg.getInt("DiskBuffer").val();
	liveStreamWriter = new ImageDiskWriter( saveFolder, diskBufferSize );
	imageReceiver.setDiskWriter( liveStreamWriter );
	
	// start the network receiver
	imageReceiver.startReceiving( null, null );	
	
	// start the reconstruction threads
	reconRunner = new ReconstructionRunner(cfg, avf, channels); 

	// start the SIM sequence detection
	seqDetection = new SimSequenceExtractor(cfg, imageReceiver, reconRunner, this);

	// setup the displays
	wfDisplay    = new PlainImageDisplay( nrCh,   size,   size, channels );
	reconDisplay = new PlainImageDisplay( nrCh, 2*size, 2*size, channels );
	JFrame hrFr = new JFrame("Reconstruction");
	JFrame lrFr = new JFrame("Widefiled");
	hrFr.add( reconDisplay.getPanel() );
	lrFr.add(    wfDisplay.getPanel() );
	hrFr.pack();
	lrFr.pack();

	hrFr.setVisible(true);
	lrFr.setVisible(true);

	// setup main interface tabs
	JTabbedPane tabbedPane = new JTabbedPane();
    
	tabbedPane.addTab( "main", mainPanel );
        tabbedPane.addTab("controller", new ClientGui(cfg, channels, seqDetection, reconRunner) );

	for (int ch=0 ; ch<nrCh ; ch++) {
	    ParameterTab pTab = new ParameterTab( reconRunner, ch, cfg );
	    tabbedPane.addTab( channels[ch], pTab.getPanel());
	}
	    

	mainFrame.add(tabbedPane);
	mainFrame.pack();
        mainFrame.setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
	mainFrame.setVisible(true);
	
	// setup the display-update threads
	SimpleImageForward sif1 = new SimpleImageForward(false);
	SimpleImageForward sif2 = new SimpleImageForward(true);

	sif1.start();
	sif2.start(); 


	DynamicDisplayUpdate updateThread = new DynamicDisplayUpdate();
	updateThread.start();

    }


   
    /** Thread updating dynamic display */
    class DynamicDisplayUpdate extends Thread {
    
	public void run() {
	    while (true) {
		// update save buffer state
		fileBufferBar.setString( String.format("%7.0f MB / %7.0f sec left",
		    liveStreamWriter.getSpace()/1024/1024.,
		    liveStreamWriter.getTimeLeft(512, 1, 100) ));
		fileBufferBar.setValue( liveStreamWriter.bufferState());
	
		int dropped = liveStreamWriter.nrDroppedFrames();
		if ( dropped > 0 && isRecording ) Tool.error("#"+dropped+" not saved", false);

		try {
		    Thread.sleep(500);
		} catch (InterruptedException e) {
		    return;
		}	
	    }
	}

    }


    class SimpleImageForward extends Thread {
	
	final boolean doWidefield;
	
	SimpleImageForward( boolean dwf ) {
	    doWidefield = dwf;
	}

	public void run() {
	    while (true) {
		try {
		    Vec2d.Real [] img ; 
		    if (doWidefield) {
		        img = reconRunner.finalWidefield.take();
		    } else {
		        //reconRunner.finalRecon.take();
		        //reconRunner.finalRecon.take();
		        //reconRunner.finalRecon.take();
		        img =  reconRunner.finalRecon.take();
		    }
		    
		    for (int c=0; c<reconRunner.nrChannels; c++) {
			if (doWidefield)
			    wfDisplay.newImage( c, img[c] );
			else
			    reconDisplay.newImage( c, img[c] );
		    }
		} catch (InterruptedException e ) {
		    Tool.trace("Display thread interrupted, why?");
		}
		    
		if (doWidefield)
		    wfDisplay.refresh();
		else
		    reconDisplay.refresh();
	    }
	}
    }

    /** starts and displays the GUI */
    public static void main(String [] arg) {
	// tweak SimpleMT
	SimpleMT.setNrThreads( Math.max( SimpleMT.getNrThreads()-2, 2));
	
	
	// load the CUDA library
        String OS = System.getProperty("os.name").toLowerCase();
        VectorFactory avf;
        // following Factory for Linux-GPU-Reconstruction
        if ( OS.contains("nix") || OS.contains("nux") || OS.contains("aix") ) {
            String wd = System.getProperty("user.dir")+"/accel/";
            Tool.trace("loading library from: "+wd);
            System.load(wd+"libcudaimpl.so");
            avf = AccelVectorFactory.getFactory();
        }
        
        // following Factory for Windows-GPU-Reconstruction
        else if ( OS.contains("win") ) {
            String wd = System.getProperty("user.dir")+"\\";
            Tool.trace("loading library from: "+wd);
            System.load(wd+"libcudaimpl.dll");
            avf = AccelVectorFactory.getFactory();
        }
        
        // following Factory for CPU-Reconstruction
        else {
            avf = Vec.getBasicVectorFactory();
        }
        
        if (arg.length<2) {
	    System.out.println("Start with: config-file.xml [488] [568] [647] ...");
	    return;
	}
	try {
	    Conf cfg = Conf.loadFile( arg[0] );
	    LiveControlPanel lcp = new LiveControlPanel( 
		cfg.r().cd("vigor-settings"),avf, 
		Arrays.copyOfRange(arg, 1, arg.length));
	} catch (Exception e) {
	    System.err.println("Error loading config or initializing");
	    e.printStackTrace();
	    System.exit(-1);
	}
    }



}
