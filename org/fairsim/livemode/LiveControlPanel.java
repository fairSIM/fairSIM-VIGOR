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

import java.awt.Color;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

import org.fairsim.utils.Conf;
import org.fairsim.transport.ImageReceiver;
import org.fairsim.transport.ImageDiskWriter;
import org.fairsim.transport.ImageWrapper;

import org.fairsim.utils.Tool;

/** Provides the control interface for live mode */
public class LiveControlPanel {

    boolean isRecording = false;    // if the raw stream is recorded

    final JProgressBar fileBufferBar;

    final ImageDiskWriter  liveStreamWriter;
    final ImageReceiver	   imageReceiver;

    final JTextArea  statusField;
    final JTextField statusMessage;

    public LiveControlPanel(final Conf.Folder cfg) 
	throws Conf.EntryNotFoundException, java.io.IOException {
	
	// get parameters
	final int imgSize = cfg.getInt("NetworkBuffer").val();


	// initialize network receiver
	imageReceiver    = new ImageReceiver(50,512,512);
	imageReceiver.addListener( new ImageReceiver.Notify() {
	    public void message( String m , boolean err, boolean fail) {
	        String e = (err)?( (fail)?("FAIL: "):("err: ") ):("net: ");
		statusField.append(e);
	    }
	});

	// initialize disk writer
	String saveFolder = cfg.getStr( "DiskFolder" ).val();
	int bufferSize = cfg.getInt("DiskBuffer").val();
	liveStreamWriter = new ImageDiskWriter( saveFolder, bufferSize );

	//  ------- 
	//  initialize the GUI
	//  ------- 
	
	JFrame mainFrame = new JFrame("Live SIM control");
	JPanel mainPanel = new JPanel();
	mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));

	// GUI - image record function
	JPanel recorderPanel = new JPanel();
	recorderPanel.setBorder(BorderFactory.createTitledBorder(
	    "raw Stream recording") );
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

	fileBufferBar = new JProgressBar();
	fileBufferBar.setString("save buffer");
	fileBufferBar.setStringPainted(true);

	recorderPanel.add( recordButton );
	recorderPanel.add(fileBufferBar);
	mainPanel.add(recorderPanel);



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
	statusField = new JTextArea(20,30);
	statusField.setEditable(false);
	DefaultCaret cr = (DefaultCaret)statusField.getCaret();
	cr.setUpdatePolicy( DefaultCaret.ALWAYS_UPDATE );
	
	statusPanel.add(statusField);
	mainPanel.add(statusPanel);
	
	statusMessage = new JTextField(30);
	statusMessage.setEditable(false);

	mainPanel.add(statusMessage);

	// redirect log output
	Tool.setLogger( new Tool.Logger() {
	    public void writeTrace( String e ) {
		statusField.append(e);
	    }
	    public void writeShortMessage( String e) {
		statusMessage.setText(e);
	    }
	});
    
	// setup frame and display
	mainFrame.add(mainPanel);
	mainFrame.pack();
	mainFrame.setVisible(true);
	
	// start a thread updating the display


    }


    /** starts and displays the GUI */
    public static void main(String [] arg) {
	try {
	    Conf cfg = Conf.loadFile( arg[0] );
	    LiveControlPanel lcp = new LiveControlPanel( cfg.r().cd("vigor-settings"));
	} catch (Exception e) {
	    System.err.println("Error loading config or initializing");
	    e.printStackTrace();
	    return;
	}
    }


    /** Thread updating dynamic display */
    class DynamicDisplayUpdate extends Thread {
    
	public void run() {
	    // update save buffer state



	}

    }



}
