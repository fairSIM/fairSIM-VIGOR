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
import javax.swing.JButton;
import javax.swing.BoxLayout;

import java.awt.GridLayout;
import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;

import javax.swing.BorderFactory;
import javax.swing.JProgressBar;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.JScrollPane;
import javax.swing.JCheckBox;

import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

import java.util.Arrays;

import org.fairsim.utils.Conf;
import org.fairsim.sim_algorithm.SimParam;
import org.fairsim.sim_algorithm.OtfProvider;
//import org.fairsim.transport.ImageReceiver;
//import org.fairsim.transport.ImageDiskWriter;
//import org.fairsim.transport.ImageWrapper;
//import org.fairsim.linalg.VectorFactory;
//import org.fairsim.accel.AccelVectorFactory;

import org.fairsim.utils.Tool;
import org.fairsim.sim_gui.Tiles;

/** Provides a paramter overview for each channel */
public class ParameterTab {

    boolean isRecording = false;    // if the raw stream is recorded

    final JPanel mainPanel = new JPanel();
	
    private FilterParameters fp = new FilterParameters();
    private ReconParameters rp ;

    final private SimParam sp;	// the SimParam object we work on

    ParameterTab( Conf.Folder cfg, SimParam sp ) {
	if (sp==null)	// TODO: throw null pointer here
	    throw new RuntimeException("Null pointer");
	this.sp = sp;
	rp = new ReconParameters();

	mainPanel.add( fp.panel );
	mainPanel.add( rp.panel );
    }
   
    
    /** sub-class for Wiener filter parameters */
    class FilterParameters {
	JPanel panel = new JPanel();
	{
	    Tiles.ValueSlider wienerParam = 
		new Tiles.ValueSlider(0, 0.25, 0.005, 0.05);
	    Tiles.ValueSlider attStr = 
		new Tiles.ValueSlider(0.5, 1, 0.0025, 0.9);
	    Tiles.ValueSlider attFWHM = 
		new Tiles.ValueSlider(0.5, 2.5, 0.005, 1.2);

	    JCheckBox attenuationCheckbox = 
		new JCheckBox("Enable attenuation");

	    JButton applyButton = new JButton("apply!");

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
    class ReconParameters {
    
	JPanel panel = new JPanel();

	{
	    
	    JTextArea statusField = new JTextArea(30,10);
	    

	}
    }



    public JPanel getPanel() {
	return mainPanel;
    }


    /** Small test display function */
    public static void main( String [] arg) throws Exception {
	    Conf cfg = Conf.loadFile( arg[0] );
	    Conf.Folder fld = cfg.r().cd("vigor-settings").cd("channel-"+arg[1]);
	    SimParam param = SimParam.loadConfig( fld );
	    JFrame jf = new JFrame("Test display");
	    ParameterTab pt = new ParameterTab( cfg.r().cd("vigor-settings"), param);
	    jf.add( pt.getPanel() );
	    jf.pack();
	    jf.setVisible(true);
    }

}
