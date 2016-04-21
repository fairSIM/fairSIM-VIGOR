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

package org.fairsim.fiji;

import org.fairsim.utils.Tool;
import org.fairsim.utils.ImageDisplay;

import org.fairsim.sim_gui.FairSimGUI;
import org.fairsim.sim_algorithm.SimParam;

import javax.swing.JFrame;
import javax.swing.JOptionPane;

import java.util.Scanner;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.BufferedReader;

import ij.plugin.PlugIn;
import ij.IJ;
import ij.ImagePlus;


public class FairSim_ImageJplugin implements PlugIn {

    /** Called by Fiji to start the plugin */
    public void run(String inputarg) {
	
	SimParam sp=null;
	String [] args = inputarg.split("-");

	// amount and redirection of output
	if ( args.length>1 && args[1].equals("log")) 
	    setLog(true);
	else
	    setLog(false);

	// show the 'about' window
	if (args[0].equals("about")) {
	    showAbout();
	    return;
	}


	boolean fromFile = false;

	// create new reconstruction, or load
	if (args[0].equals("new"))
	    sp = FairSimGUI.newSpDialog( IJ.getInstance());

	if (args[0].equals("load")) {
	    try {
		sp = FairSimGUI.fromFileChooser( IJ.getInstance() );
		fromFile = true;
	    } catch (Exception e ) {
		JOptionPane.showMessageDialog( IJ.getInstance(),
		 "Problem loading file\n"+e, "fairSIM laod file",
		 JOptionPane.ERROR_MESSAGE);
	    }

	}

	if (sp==null)
	    return;

	// create the main GUI
	FairSimGUI a =  new FairSimGUI( 
	    sp,
	    new ImageOpener(),
	    DisplayWrapper.getFactory(),
	    fromFile
	    );
	    
    }

    /** set the logger (and amount) */
    void setLog(boolean full) {
	if (full) {
	    Tool.setLogger( new Tool.Logger () {
		@Override
		public void writeTrace(String w) {
		    IJ.log(w);
		}
		@Override
		public void writeShortMessage(String w) {
		    IJ.showStatus(w);
		}
		@Override
		public void writeError(String w, boolean fatal) {
		    IJ.log( (fatal)?("FATAL ERROR: "):("error") + w );
		}

	    });
	    Tool.trace("fairSIM started with log output");
	} else {
	    Tool.setLogger( new Tool.Logger () {
		@Override
		public void writeTrace(String w) {
		
		}
		@Override
		public void writeShortMessage(String w) {
		    IJ.showStatus(w);
		}
		@Override
		public void writeError(String w, boolean fatal) {
		    IJ.log( (fatal)?("FATAL ERROR: "):("error") + w );
		}
	    });

	}
    }

    /** open the 'about' window */
    void showAbout() {

	InputStream is1 = getClass().getResourceAsStream("/org/fairsim/resources/about.html");
	InputStream is2 = getClass().getResourceAsStream("/org/fairsim/git-version.txt");
	if ( is1 == null ) {
		JOptionPane.showMessageDialog( IJ.getInstance(),
		 "About information not found", "about fairSIM",
		 JOptionPane.WARNING_MESSAGE);
		return;
	}   

	// get the about text	
	BufferedReader br = new BufferedReader( new InputStreamReader( is1 ) );
	StringBuffer text = new StringBuffer();
	String line;
	try {
	    while ( (line = br.readLine()) != null )
		text.append( line );
	} catch ( java.io.IOException e ) {
	    text = new StringBuffer("failed to read about information"); 
	}

	// get the version information
	String gitCommit = "not found";
	String version   = "not found";
	if ( is2 != null ) {
	    BufferedReader br2 = new BufferedReader( new InputStreamReader( is2 ) );
	    try {
		gitCommit = br2.readLine();
		version   = br2.readLine();
	    } catch ( java.io.IOException e ) {
		gitCommit = "n/a";   
		version = "unknown";
	    }
	}


	//String text = new Scanner( is, "UTF-8" ).useDelimiter("\\A").next();

	JOptionPane.showMessageDialog( IJ.getInstance(),
	    "<html>"+text+"<br /><p>"+
	    "version: "+version.substring(0, Math.min(12, version.length()))+
	    "<br />git build id: "+
	    gitCommit.substring(0, Math.min(10, gitCommit.length()))+
	    "</html>", "About fairSIM",
	    JOptionPane.INFORMATION_MESSAGE);
    
    }





    /** for testing */
    public static void main( String [] arg ) {

	if (arg.length<1)
	    return;
	
	ImagePlus ip = IJ.openImage(arg[0]);
	ip.show();

	FairSimGUI a =  new FairSimGUI( 
	    SimParam.create(3,3,5,512, 0.082, null),
	    new ImageOpener(),
	    DisplayWrapper.getFactory(),
	    false
	    );
    }


}
