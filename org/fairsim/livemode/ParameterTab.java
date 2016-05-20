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

import javax.swing.JPanel;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
	
import javax.swing.JButton;
import javax.swing.BoxLayout;
import javax.swing.Box;

import java.awt.GridLayout;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.Insets;

import javax.swing.BorderFactory;
import javax.swing.JProgressBar;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.JScrollPane;
import javax.swing.JCheckBox;
import javax.swing.AbstractListModel;

import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

import java.io.File;
import java.io.FileNotFoundException;

import org.fairsim.utils.Conf;
import org.fairsim.sim_algorithm.SimParam;
import org.fairsim.sim_algorithm.OtfProvider;
//import org.fairsim.transport.ImageReceiver;
//import org.fairsim.transport.ImageDiskWriter;
//import org.fairsim.transport.ImageWrapper;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.linalg.Vec;
//import org.fairsim.accel.AccelVectorFactory;

import org.fairsim.utils.Tool;
import org.fairsim.sim_gui.Tiles;

/** Provides a paramter overview for each channel */
public class ParameterTab {

    boolean isRecording = false;    // if the raw stream is recorded

    final JPanel mainPanel = new JPanel();
	
    private FilterParameters fp ;
    private ReconParameters rp ;

    final private int ourChannelIndex;
    final private ReconstructionRunner.PerChannel ourChannel;
    final private ReconstructionRunner		  ourReconRunner;

    final File paramDir;

    /** Create a parameter tab linked to the (live) reconstruction channel c */
    ParameterTab( ReconstructionRunner rr, int chIdx, Conf.Folder cfg ) 
	throws Conf.EntryNotFoundException, FileNotFoundException {

	this.ourChannelIndex = chIdx;
	this.ourReconRunner = rr;
	this.ourChannel = rr.getChannel(chIdx);
	
	paramDir = Tool.getFile(cfg.getStr("ParamFolder").val());
	if ( !paramDir.isDirectory()) {
	    throw new FileNotFoundException("Provided path is not a folder: "+
		paramDir.getPath());
	}
    
	fp = new FilterParameters();
	rp = new ReconParameters();

	mainPanel.setLayout( new BoxLayout(mainPanel, BoxLayout.PAGE_AXIS));

	mainPanel.add( fp.panel );
	mainPanel.add( rp.panel );
    }
   
    
    /** sub-class for Wiener filter parameters */
    class FilterParameters {
	JPanel panel = new JPanel();
	{
	    final Tiles.ValueSlider wienerParam = 
		new Tiles.ValueSlider(0, 0.25, 0.005, 0.05);
	    final Tiles.ValueSlider attStr = 
		new Tiles.ValueSlider(0.5, 1, 0.0025, 0.9);
	    final Tiles.ValueSlider attFWHM = 
		new Tiles.ValueSlider(0.5, 2.5, 0.005, 1.2);

	    final JCheckBox attenuationCheckbox = 
		new JCheckBox("Enable attenuation");

	    JButton applyButton = new JButton("apply!");

	    applyButton.addActionListener( new ActionListener() {
		public void actionPerformed(ActionEvent e) {
		    ourChannel.useAttenuation  = attenuationCheckbox.isSelected();
		    ourChannel.attStr  = attStr.getVal();
		    ourChannel.attFWHM = attFWHM.getVal();
		    ourChannel.wienParam = wienerParam.getVal();
		    boolean ok = ourReconRunner.doFilterUpdate.offer(ourChannelIndex);
		    if (!ok)
			Tool.trace("too many changes queued, please wait...");
		}
	    });
	

	    panel.setLayout(new GridBagLayout());
	    panel.setBorder(BorderFactory.createTitledBorder(
		"Filter parameters") );
	    
	    GridBagConstraints c = new GridBagConstraints();	
	    c.gridx=0; c.gridy=0; c.gridwidth=2; c.gridheight=1;
	    panel.add( new JLabel("Wiener filter"),c);
	    c.gridy=2; 
	    panel.add( new JLabel("Att. str."),c);
	    c.gridy=3;
	    panel.add( new JLabel("Att. FWHM"),c);

	    c.gridx=2; c.gridy=0; c.gridwidth=6; c.gridheight=1;
	    panel.add( wienerParam, c );
	    c.gridy=2; 
	    panel.add( attStr, c );
	    c.gridy=3;
	    panel.add( attFWHM, c );
	    
	    c.gridy=1; c.gridx=3; c.gridwidth=4;
	    panel.add( attenuationCheckbox,c );
	    c.gridy=4;
	    panel.add( applyButton, c );

	}
    }

    /** sub-class for reconstruction parameters */
    class ReconParameters implements Tool.Callback<SimParam> {
    
	JPanel panel = new JPanel();

	final Tiles.TList<SimParam> availableParameters;

	// general initialization
	{

	    // displaying the available parameter sets
	    availableParameters = new Tiles.TList<SimParam>() {
		@Override
		public String convertToString(SimParam sp, int idx) {
		    return Tool.readableTimeStampMillis( sp.getParamTimestamp(), true)+
			((idx==0)?(" (default)"):(""));
		}
	    };

	    //availableParameters.setVisibleRowCount(5);
	    JScrollPane pane1 = new JScrollPane( availableParameters,
		JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
		JScrollPane.HORIZONTAL_SCROLLBAR_AS_NEEDED);

	    availableParameters.addElement( ourChannel.param );
	    fillListFromFolder();

	    // displaying the currently active parameter
	    final JTextArea	statusField = new JTextArea(12,35);
	    statusField.setEditable(false);

	    statusField.setText(ourChannel.param.prettyPrint(true));

	    // button to set the selected parameter set as the current one
	    JButton useParameterButton = new JButton("apply parameters");
	    useParameterButton.setToolTipText(
		"use the parameter set selected in the list for reconstruction");

	    useParameterButton.addActionListener( new ActionListener() {
		public void actionPerformed(ActionEvent e) {
		    SimParam sp = availableParameters.getSelectedValue();
		    if (sp!=null) {
			ourChannel.param = sp;
			statusField.setText(ourChannel.param.prettyPrint(true));
		    }
		}

	    });



	    // merging everything in one panel
	    JButton	runFitButton = new JButton("run fit");
	    runFitButton.setToolTipText("run a parameter fit on the current image");

	    panel.setLayout(new GridBagLayout());
	    panel.setBorder(BorderFactory.createTitledBorder(
		"Reconstruction parameters") );
	   
	    GridBagConstraints c = new GridBagConstraints();
	    //c.insets = new Insets(3,3,3,3);
	    
	    // status
	    c.gridx=0; c.gridy=0; c.gridheight=3; c.gridwidth=4; 
	    panel.add( statusField, c );
	    
	    // list
	    c.weightx = 1;
	    c.weighty = 1;
	    c.fill = GridBagConstraints.BOTH;
	    c.gridx=0; c.gridy=3; c.gridheight=7; c.gridwidth=4;  
	    JPanel tmp1 = new JPanel();
	    tmp1.add( pane1 );
	    panel.add( tmp1, c );
	  
	    // buttons
	    c.weightx=0; c.weighty=0;
	    c.gridx=1; c.gridy=10; c.gridheight=1; c.gridwidth=4; 
	    JPanel buttonsPanel = new JPanel();
	    buttonsPanel.add( useParameterButton );
	    buttonsPanel.add( runFitButton );
	    panel.add( buttonsPanel, c);
    
	    
	    runFitButton.addActionListener( new ActionListener() {
		public void actionPerformed(ActionEvent e) {
		    
		    Tool.Tuple<Integer, Tool.Callback<SimParam>> errant=
			new Tool.Tuple<Integer, Tool.Callback<SimParam>>(
			    ourChannelIndex, ReconParameters.this  );
		    
		    boolean ok = ourReconRunner.doParameterRefit.offer(errant);
		    if (!ok)
			Tool.trace("too many updates pending, please wait...");
		}
	    });
	

	}
   
	@Override
	public void callback(SimParam sp) {
	    //Tool.trace("new computation done, new param:");
	    //Tool.trace(sp.prettyPrint(true));
	    availableParameters.addElement(sp);
	    String name = "param_ch"+ourChannel.chNumber+"_"+
		Tool.readableTimeStampMillis( sp.getParamTimestamp(), false)+".xml";
	    Tool.trace("Storing new parameter as: "+name);
	    // store the result
	    Conf paramConf = new Conf("fairsim");
	    sp.saveConfig( paramConf.r());
	    try {
		paramConf.saveFile( paramDir.getPath()+File.separator+name);
	    } catch (Conf.SomeIOException e) {
		Tool.error("Saving failed: "+e, false);
	    }
	}
    
	/** scans a folder and loads all parameter sets matching a pattern */
	void fillListFromFolder() {
	    File [] listOfFiles = paramDir.listFiles();
	    for ( File i : listOfFiles ) {
		if (i.getName().startsWith("param_ch"+ourChannel.chNumber)) {
		    try {
			Conf cfg = Conf.loadFile( i.getPath());
			SimParam sp = SimParam.loadConfig( cfg.r());
			availableParameters.addElement( sp );
		    } catch ( Exception e ) {
			Tool.error("Loading "+i.getName()+"failed: "+e, false);
		    }

		}
	    }

	}


    }



    public JPanel getPanel() {
	return mainPanel;
    }


    /** Small test display function */
    public static void main( String [] arg) throws Exception {
	    Conf cfg = Conf.loadFile( arg[0] );
	   
	    ReconstructionRunner rr = new ReconstructionRunner( 
		cfg.r().cd("vigor-settings"), 
		Vec.getBasicVectorFactory(), 
		new String [] {arg[1]},
		    true);

	    JFrame jf = new JFrame("Test display");
	    
	    ParameterTab pt = new ParameterTab( rr, 0, cfg.r().cd("vigor-settings") );
	    jf.add( pt.getPanel() );
	    jf.pack();
	    jf.setVisible(true);
   
	    // If a TIF was provided, send it
	    if (arg.length>2) {
		ij.ImagePlus ip = ij.IJ.openImage(arg[2]);
		ij.ImageStack iSt = ip.getImageStack();
		short [][][] imgs = new 
		    short[1][rr.nrDirs * rr.nrPhases][rr.width*rr.height];
		int stackPos=0;
		for (int a=0; a<rr.nrDirs; a++)
		    for (int p=0; p<rr.nrPhases; p++) {
			ij.process.ShortProcessor sp = 
			    iSt.getProcessor(stackPos+1).convertToShortProcessor();

			System.arraycopy( (short[])sp.getPixels(),0,  
			    imgs[0][stackPos],
			    0, rr.width * rr.height);
			stackPos++;
		}
		rr.queueImage( imgs );
	    }
    
    }

}
