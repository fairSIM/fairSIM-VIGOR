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

package org.fairsim.utils;

import javax.swing.JFrame;
import javax.swing.JButton;
import javax.swing.JPanel;

import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

/**
 * Displays a frame with a stop button.
 * Handy for use in (MicroManager) scripts.
 * On click, the button vanishes, and its state changes to "stop"
 * */
public final class StopButton {

    private boolean hasBeenPressed = false;

    public StopButton() {
	this("");
    }

    public StopButton(String extraText) {
	
	final JFrame mainFrame = new JFrame(extraText);
	final JPanel mainPanel = new JPanel();
	JButton stopButton = new JButton("stop!"+extraText);

	mainPanel.add(stopButton);
	mainFrame.add(mainPanel);
	mainFrame.pack();
	mainFrame.setVisible(true);

	mainFrame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);

	stopButton.addActionListener( new ActionListener() {
	    public void actionPerformed(ActionEvent e) {
		hasBeenPressed = true;
		mainFrame.setVisible(false);
		mainFrame.dispose();
	    }
	});


    }

    public boolean state() {
	return hasBeenPressed;
    }


    public static void main( String [] arg ) 
	throws InterruptedException {
	StopButton sb = new StopButton();

	while ( true ) {
	    System.out.println("State: " + sb.state());
	    Thread.sleep(2500);
	}
    }


}



