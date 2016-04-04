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

package org.fairsim.sim_gui;

import javax.swing.JFrame;
import javax.swing.JComponent;
import javax.swing.JPanel;

import java.awt.Dimension;
import java.awt.Graphics;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import javax.swing.JSlider;
import javax.swing.JSpinner;
import javax.swing.JLabel;
import javax.swing.JButton;

import java.awt.GridBagLayout;
import javax.swing.BoxLayout;
import java.awt.GridBagConstraints;

import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

import org.fairsim.network.ImageReceiver;
import org.fairsim.network.ImageWrapper;

import org.fairsim.linalg.Vec2d;

import org.fairsim.utils.Tool;

public class PlainImageDisplay {

    private final JPanel mainPanel ;
    private final ImageComponent ic ;

    public PlainImageDisplay(int w, int h) {

	ic = new ImageComponent(w,h);
	mainPanel = new JPanel();
	mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));

	// panel containing the image
	JPanel p1 = new JPanel();
	p1.add(ic);
	mainPanel.add( p1 );

	// sliders and buttons
	final JSlider sMin = new JSlider(JSlider.HORIZONTAL, 0, 16*100, 0);
	final JSlider sMax = new JSlider(JSlider.HORIZONTAL, 200, 16*100, 12*100);
	final JButton autoMin = new JButton("auto");
	final JButton autoMax = new JButton("auto");
	final JLabel  valMin = new JLabel( String.format("% 5d",sMin.getValue()));
	final JLabel  valMax = new JLabel( String.format("% 5d",sMax.getValue()));
	
	final JSlider sGamma = new JSlider(JSlider.HORIZONTAL, 10, 300,100);
	final JLabel  lGamma = new JLabel(String.format("g%4.2f", sGamma.getValue()/100.));
	final JButton bGamma1 = new JButton("1.0");
	final JButton bGamma2 = new JButton("2.2");

	sMin.addChangeListener( new ChangeListener() {
	    public void stateChanged(ChangeEvent e) {
		double exponent = sMin.getValue()/100.;
		int val = (int)Math.pow( 2, exponent );

		if (sMax.getValue() -200 < exponent*100 )
		    sMax.setValue( (int)(exponent*100)+200 );
		
		valMin.setText(String.format("2^%4.2f -> % 5d",exponent, val));
		ic.scalMin = val;
		ic.paintImage();
	    }
	});

	sMax.addChangeListener( new ChangeListener() {
	    public void stateChanged(ChangeEvent e) {
		
		double exponent = sMax.getValue()/100.;
		int val = (int)Math.pow( 2, exponent );

		if (sMin.getValue() +200 > exponent*100 )
		    sMin.setValue( (int)(exponent*100)-200 );
		
		valMax.setText(String.format("2^%4.2f -> % 5d",exponent, val));
		
		//int val = sMax.getValue();
		//if (sMin.getValue()+9>val)
		//    sMin.setValue(val-10);
		//valMax.setText(String.format("% 5d",val));
		
		ic.scalMax = val;
		ic.paintImage();
	    }
	});

	autoMin.addActionListener( new ActionListener() {
	    public void actionPerformed( ActionEvent e ) {
		sMin.setValue( (int)(Math.log( ic.currentImgMin )*100/Math.log(2)) );
	    }
	});

	autoMax.addActionListener( new ActionListener() {
	    public void actionPerformed( ActionEvent e ) {
		//sMax.setValue( ic.currentImgMax );
		sMax.setValue( (int)(Math.log( ic.currentImgMax )*100/Math.log(2)) );
	    }
	});

	sGamma.addChangeListener( new ChangeListener() {
	    public void stateChanged(ChangeEvent e) {
		double gamma = sGamma.getValue()/100.;
		lGamma.setText(String.format("g%4.2f",gamma));
		ic.recalcGammaTable( gamma );
		ic.paintImage();
	    }
	});
	
	bGamma1.addActionListener( new ActionListener() {
	    public void actionPerformed( ActionEvent e ) {
		sGamma.setValue( 100 );
	    }
	});
	bGamma2.addActionListener( new ActionListener() {
	    public void actionPerformed( ActionEvent e ) {
		sGamma.setValue( 220 );
	    }
	});

	// sliders setting min/max
	JPanel sliders = new JPanel(new GridBagLayout());
	GridBagConstraints c = new GridBagConstraints();	
	
	c.gridx=0; c.gridy=0; c.gridwidth=1; c.gridheight=1;
	sliders.add( valMin,c );
	c.gridy=1;
	sliders.add( valMax,c);
	c.gridx=1; c.gridy=0; c.gridwidth=6; c.gridheight=1;
	sliders.add( sMin, c );
	c.gridy=1;
	sliders.add( sMax, c );
	c.gridx=7; c.gridy=0; c.gridwidth=2;
	sliders.add( autoMin, c );
	c.gridy=1;
	sliders.add( autoMax, c );
    
	c.gridx=0; c.gridy=2; c.gridwidth=1;
	sliders.add( lGamma , c);
	c.gridx=1; c.gridwidth=6;
	sliders.add( sGamma , c);
	c.gridx=7; c.gridwidth=1;
	sliders.add( bGamma1 ,c );
	c.gridx=8; 
	sliders.add( bGamma2 ,c );

	mainPanel.add(sliders);

    }
  
    /** Set a new image */
    public void newImage( float [] data, int w, int h) {
	ic.setImage( data, w, h);	
    }
    
    /** Set a new image */
    public void newImage( Vec2d.Real img ) {
	ic.setImage( img);	
    }


    /** Return the GUI panel for the component */
    public JPanel getPanel() {
	return mainPanel;
    }


    /** Internal class for the actual image display */
    private static class ImageComponent extends JComponent{
         
        BufferedImage bufferedImage = null;
        //Dimension myDimension = new Dimension(512, 512);
	final int maxWidth, maxHeight;
    
	final int gammaLookupTableSize = 1024;
	
	int curWidth, curHeight;
	int scalMax=256, scalMin=0;
	int currentImgMin = 0, currentImgMax=1;
	double gamma=1.0;

	final float [] imgBuffer ;
	final byte  [] imgData   ;
	final byte  [] imgDataBuffer ;
	final byte  [] gammaLookupTable = new byte[ gammaLookupTableSize ];

        public ImageComponent(int w, int h) {
	    maxWidth=w; maxHeight=h;
	    bufferedImage = new BufferedImage(maxWidth,maxHeight, BufferedImage.TYPE_BYTE_GRAY);
	    imgBuffer = new float[w*h];
	    imgDataBuffer = new byte[w*h];
	    imgData = ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
	    recalcGammaTable(1);
	}
        
	public void setImage( float [] img, int w, int h ) {
	    if (w>maxWidth || h>maxHeight) 
		throw new RuntimeException("Image bigger than buffer");
	    curWidth=w; curHeight=h;
	    System.arraycopy( img, 0, imgBuffer, 0, w*h);
	    paintImage();
	}
	
	public void setImage( Vec2d.Real img ) {
	    if (img.vectorWidth()>maxWidth || img.vectorHeight()>maxHeight) 
		throw new RuntimeException("Image bigger than buffer");
	    curWidth=img.vectorWidth(); curHeight=img.vectorHeight();
	    float [] dat = img.vectorData();

	    for (int i=0; i<curWidth*curHeight; i++) {
		int val = (int)dat[i];
		if (val<0) val=0;
		if (val>65535) val = 65535;
		if (val>32768) val-= 65536;

		imgBuffer[i] = (short)val;

	    }

	    paintImage();
	}

	public void recalcGammaTable( double gamma ) {
	    this.gamma = gamma;
	    for (int i=0; i<gammaLookupTableSize; i++) {
		gammaLookupTable[i] = (byte)(256*Math.pow( 
		    1.*i / gammaLookupTableSize, gamma ));
	    }
	}


	public void paintImage() {

	    for (int y=0; y<curWidth; y++)
	    for (int x=0; x<curHeight; x++) {
		// scale
		float val = imgBuffer[ x + y*curWidth ];
		if (val> currentImgMax) currentImgMax = (int)val;
		if (val< currentImgMin) currentImgMin = (int)val;
		float out=0;
		if ( val >= scalMax ) out=1;
		if ( val <  scalMin ) out=0;
		if ( val >= scalMin && val < scalMax ) 
		    out = 1.f*(val - scalMin) / (scalMax-scalMin) ;
		
		//out = Math.pow( out, gamma );
		//imgDataBuffer[x + y*curWidth ] = (byte)(255.999*out); 
		imgDataBuffer[ x + y*curWidth ] = 
		    gammaLookupTable[ (int)(out*(gammaLookupTableSize-1)) ];

	    }
	    System.arraycopy( imgDataBuffer, 0 , imgData, 0, curWidth*curHeight);
	    this.repaint();
	}

 
        @Override
        public Dimension getPreferredSize() {
            return new Dimension(maxWidth, maxHeight);
        }
 
        @Override
        public Dimension getMaximumSize() {
            return new Dimension(maxWidth, maxHeight);
        }
 
        @Override
        public Dimension getMinimumSize() {
            return new Dimension(maxWidth, maxHeight);
        }
 
        @Override
        protected void paintComponent(Graphics g) {
            g.drawImage(bufferedImage, 0, 0, null);
        }
    }


    /** Main method for easy testing */
    public static void main( String [] arg ) throws java.io.IOException {
	
	
	// create an ImageDisplay sized 512x512
	PlainImageDisplay pd = new PlainImageDisplay(512,512);

	// create a frame and add the display
	JFrame mainFrame = new JFrame("Plain Image Receiver");
	mainFrame.add( pd.getPanel() ); 
	mainFrame.pack();
	mainFrame.setVisible(true);
    
	// create a network receiver
	ImageReceiver ir = new ImageReceiver(64,512,512);
	
	ir.addListener( new ImageReceiver.Notify() {
	    public void message(String m, boolean err, boolean fatal) {
		if (err || fatal) {
		    System.err.println((fatal)?("FATAL"):("Error")+" "+m);
		} else {
		    System.out.println("Recvr: "+m);
		}
	    }
	});
	
	// start receiving
	ir.startReceiving( null, null);
	int count=0; 
	double max=0;
	
	Vec2d.Real imgVec = Vec2d.createReal( 512, 512);
	
	while ( true ) {
	    ImageWrapper iw = ir.takeImage();
	   
	    iw.writeToVector( imgVec );

	    ir.recycleWrapper( iw );

	    double avr = imgVec.sumElements();
	    avr /= (iw.width() * iw.height());

	    max = Math.max(avr,max);

	    count++;
	    if (count%250==0) {
		Tool.trace("max avr pxl val: "+max);
           	max=0;
	    }
	
	    if (iw!=null)
		pd.newImage( imgVec );
	}   
	
    }


}
