/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fairsim.sim_gui;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import org.fairsim.utils.Tool;

/**
 *
 * @author Mario
 */
public class RawImageDisplay extends PlainImageDisplay {
    
    int frame;
    
    public RawImageDisplay(int frames, int nrChannels, int w, int h, String... names) {
        super(nrChannels, w, h, names);
        frame = 0;
        JLabel rawLabel = new JLabel("Raw Frame: " + frame);
        JSlider rawSlider = new JSlider(JSlider.HORIZONTAL, 0, frames, 0);
        rawSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                frame = rawSlider.getValue();
                rawLabel.setText("Raw Frame: " + frame);
                ic.paintImage();
            }
        });
        
        JPanel p = new JPanel();
        p.setLayout(new BoxLayout(p, BoxLayout.X_AXIS));
        p.add(Box.createHorizontalStrut(20));
        p.add(rawLabel);
        p.add(Box.createHorizontalStrut(20));
        p.add(rawSlider);
        p.add(Box.createHorizontalStrut(20));
        
        mainPanel.add(p, 1);
        mainPanel.add(Box.createVerticalStrut(20) , 1);
        
    }
    
    public static void main(String[] args) {
        final int size = 512;
	final int nrCh = 3;
	final int width=size, height=size;

	// create an ImageDisplay sized 512x512
	RawImageDisplay pd = new RawImageDisplay(10, nrCh, width,height);

	// create a frame and add the display
	JFrame mainFrame = new JFrame("Plain Image Receiver");
	mainFrame.add( pd.getPanel() ); 
	
	mainFrame.pack();
	mainFrame.setVisible(true);

	short [][] pxl = new short[100][width*height];

	Tool.Timer t1 = Tool.getTimer();

	for (int ch = 0; ch < nrCh; ch++) 
	for (int i=0;i<100;i++) {
	    for (int y=0;y<height;y++)
	    for (int x=0;x<width;x++) {
		if ( (x<200 || x>220) && (y<150 || y>170) )
		    pxl[i][x+y*width]=(short)(Math.random()*2500);
		else
		    pxl[i][x+y*width]=(short)(Math.random()*250);
	    }
	}

	while (true) {
	    t1.start();
	    for (int i=0;i<100;i++) {
		for (int ch = 0; ch < nrCh; ch++) {
		    pd.newImage(ch, pxl[(int)(Math.random()*99)]);
		}
		pd.refresh();
	    }
	    t1.stop();
	    System.out.println( "fps: "+((1000*100)/t1.msElapsed()) );
	}
    }
    
}
