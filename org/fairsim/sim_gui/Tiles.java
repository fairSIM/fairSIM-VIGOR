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

import java.util.List;
import java.util.ArrayList;

import javax.swing.JPanel;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.JComboBox;
import javax.swing.DefaultListCellRenderer;
import javax.swing.JList;
import javax.swing.ListModel;
import javax.swing.DefaultListModel;
import javax.swing.ListCellRenderer;
import javax.swing.JComponent;
import javax.swing.JSlider;
import javax.swing.AbstractListModel;

import javax.swing.BoxLayout;
import javax.swing.Box;
import java.awt.Component;

import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import javax.swing.event.ChangeListener;
import javax.swing.event.ChangeEvent;
import java.awt.Dimension;


/** Various GUI components */
public class Tiles {

    /** Components that allow the user to select a number */
    public interface NumberTile {
	/** Retrieves the currently display value */
	public double getVal();
    }

    /** Labeled JSpinner to select numbers */
    public static class LNSpinner extends JPanel implements NumberTile {

	private List<NumberListener> listener = new ArrayList<NumberListener>();
    
	/** Access to the spinner */
	final public JSpinner spr; 
	
	/** Create a spinner
	 *  @param label Label text in front of spinner 
	 *  @param start Initial value
	 *  @param min	 Minmal value
	 *  @param max	 Maximal value
	 *  @param inc   Increment */
	public LNSpinner( String label, double start, double min, double max, double inc ) { 
	    super();
	    super.setLayout( new BoxLayout(this, BoxLayout.LINE_AXIS));
	    
	    final JLabel jl = new JLabel(label);
	    final LNSpinner ref = this;

	    spr = new JSpinner( new SpinnerNumberModel(start,min,max,inc));
	    spr.setMaximumSize( spr.getPreferredSize() );;
	    //spr.setEditor( new JSpinner.NumberEditor( spr, "##0.00"));

	    spr.addChangeListener( new ChangeListener() {
	        public void stateChanged( ChangeEvent e ) {
		   for ( NumberListener i : listener ) {
			i.number( ref.getVal(), ref );
		   }
		}
	    }); 
	    
	    super.add( Box.createRigidArea(new Dimension(5,0)));
	    super.add( Box.createHorizontalGlue());
	    super.add( jl );
	    super.add( Box.createRigidArea(new Dimension(5,0)));
	    super.add( spr );
	    super.add( Box.createHorizontalGlue());
	    super.add( Box.createRigidArea(new Dimension(5,0)));
	}

	/** Get the spinners current value */
	@Override
	public double getVal() {
	    return ((Number)spr.getValue()).doubleValue();
	}
    
	/** Add a NumberListener */
	public void addNumberListener( NumberListener l ) {
	    listener.add( l );
	}
	
	/** Remove a NumberListener */
	public void removeNumberListener( NumberListener l ) {
	    listener.remove( l );
	}

    }

    /** Notification that an LNSpinner changed to a new number */
    public interface NumberListener {
	/** Gets called with the new number */
	public void number(double n, NumberTile e);

    }


    /** JSlider set to a defined value range and step size */
    public static class ValueSlider extends JPanel implements NumberTile {

	final double min, max, stepsize;
	final int    sliderLength;
	double value;
    
	private List<NumberListener> listener = new ArrayList<NumberListener>();
	final JSlider slider;

	public ValueSlider( double imin, double imax, double step, double initVal) {
	    if (imin>=imax || step<=0 || initVal>imax || initVal<imin )
		throw new RuntimeException("Parameters not sesible");
	    this.min = imin;
	    this.max = imax;
	    this.stepsize = step;
	    sliderLength = (int)((max-min)/stepsize);
	    int initPos  = (int)((initVal-min)/stepsize);
	    this.value   = min + initPos*stepsize;

	    final JLabel curValueLabel = new JLabel(String.format("%8.3f",value));

	    slider = new JSlider(JSlider.HORIZONTAL, 0, sliderLength, initPos);
	    final ValueSlider ref = this;

	    slider.addChangeListener( new ChangeListener() {
		public void stateChanged(ChangeEvent e) {
		    ref.value = slider.getValue()*stepsize+min;
		    curValueLabel.setText( String.format("%8.3f",ref.value));
		    for ( NumberListener l : listener )
			l.number( ref.value, ref );
		}
	    });
	    this.add(slider);
	    this.add(curValueLabel);
	}

	/** Retrieve the value the slider is currently set to */
	public double getVal() {
	    return value;
	}

	/** Add a NumberListener */
	public void addNumberListener( NumberListener l ) {
	    listener.add( l );
	}
	
	/** Remove a NumberListener */
	public void removeNumberListener( NumberListener l ) {
	    listener.remove( l );
	}
    
    
    }


   

    /** Labeled drop-down selection box */
    public static class LComboBox<T> extends JPanel {
	
	boolean suppressEvents=false;

	List<SelectListener<T>> listener 
	    = new ArrayList<SelectListener<T>>();

	/** Access to the ComboBox. */
	final public TComboBox<T> box;
	
	/** 
	 * @param label Label in front of box
	 * @param opts  Selectable elements */
	public LComboBox(String label, T ... opts ) {
	    this( label, (java.awt.Component)null, false, opts );
	}

	/** 
	 * @param label Label in front of box
	 * @param addComp Additional component, added directly after the box
	 * @param opts  Selectable elements */
	public LComboBox(String label, java.awt.Component addComp, T ... opts ) {
	    this( label, addComp, false, opts );
	}
	
	/** 
	 * @param label Label in front of box
	 * @param addComp Additional component, added directly after the box
	 * @param showToolTip if true, display the full text for each entry as tooltip
	 * @param opts  Selectable elements */
	public LComboBox(String label, java.awt.Component addComp, 
	    boolean showToolTip, T ... opts ) {
	    
	    super();
	    super.setLayout( new BoxLayout(this, BoxLayout.LINE_AXIS));
	    final JLabel jl = new JLabel(label);
	    
	    if ((opts!=null)&&(opts.length>0))
		box = new TComboBox<T>(opts);
	    else
		box = new TComboBox<T>();
	  
	    //box.setMaximumSize( box.getPreferredSize() );;

	    box.addActionListener( new ActionListener() {
	        public void actionPerformed( ActionEvent e ) {
		    if (!suppressEvents ) {
			for ( SelectListener<T> i : listener )
			    i.selected(  getSelectedItem(), box.getSelectedIndex() );
		    }
		}
	    });
	   

	    // display tooltip with full file name
	    if (showToolTip) {
		box.setRenderer( new DefaultListCellRenderer() {
		    @Override
		    public Component getListCellRendererComponent(JList list, Object value,
			int index, boolean isSelected, boolean cellHasFocus) {
			
			JComponent comp = (JComponent) super.getListCellRendererComponent(list,
			    value, index, isSelected, cellHasFocus);

			if (-1 < index && null != value ) {
			    list.setToolTipText( value.toString() );
			}
			return comp;
		    }
		});
	    }

	    super.add( Box.createHorizontalGlue());
	    super.add( jl );
	    super.add( Box.createRigidArea(new Dimension(5,0)));
	    super.add( box );
	    super.add( Box.createHorizontalGlue());
	    if (addComp!=null) {
		super.add( Box.createRigidArea(new Dimension(5,0)));
		super.add( addComp );
	    }
	}

	/** Returns the currently selected item. Might return
	 * 'null' if the list is empty. */
	@SuppressWarnings("unchecked")
	public T getSelectedItem() {
	    return (T)box.getSelectedItem();
	}

	public int getSelectedIndex() {
	    return box.getSelectedIndex();
	}

	/** Add a listener to be notified when the selection changes */
	public void addSelectListener( SelectListener<T> l ) {
	    listener.add(l);
	}
	/** Remove the listener */
	public void removeSelectListener( SelectListener<T> l ) {
	    listener.remove(l);
	}
    
	/** Fill the selector box with new elements.
	 *  If the currently selected element is contained in the new list,
	 *  it will be selected again. Otherwise, the first element is selected,
	 *  and an event is send.*/
	public void newElements( T ... opts ) {
	    newElements(-1, opts );
	}

	/** Fill the selector box with new elements, select the i'th element. */
	public void newElements( int idx, T ... opts ) {
	    
	    suppressEvents=true;

	    T curSel = getSelectedItem(); 
	    box.removeAllItems();
	    boolean newSelection=true;
	    
	    // only add new elements if opts is not empty
	    if (( opts != null ) && ( opts.length > 0)) {
		for ( T a : opts ) {
		    box.addItem( a );
		    if (( a.equals( curSel ) )&&(idx<0)) {
			box.setSelectedItem( a );
			newSelection=false;
		    }
		}
		if (idx>=0)
		    box.setSelectedIndex(idx);
	    } 
	    // set to empty 
	    else {
		    box.setSelectedIndex(-1);
	    }

	    if (newSelection)
		for ( SelectListener<T> i : listener )
		    i.selected(  getSelectedItem(), box.getSelectedIndex() );
	    
	    suppressEvents=false;
	}   

    }
   
   /** Listener to be called if things get selected */
    public interface SelectListener<T> {
	/* Selected element, its index, calling object. */
	public void selected(T e, int i); 
    } 
    
   
   /** Provides a type-save combo-box, like in java7.
     *  Wrapper around JComboBox, to fix java-1.6 to java-1.7 issue */
    public static class TComboBox<T> extends JComboBox {
	public TComboBox(T [] e) {
	    super(e);
	}
	public TComboBox() {
	    super();
	}
    };

    
    /** Provides a list model that allows to implement a presentation function */
    public static abstract class TList<T> extends JList {
	    
	public TList() {
	    super( new DefaultListModel() );
	    this.setCellRenderer( new ListCellRenderer() {
		public Component getListCellRendererComponent(
		   JList list,           // the list
		   Object value,            // value to display
		   int index,               // cell index
		   boolean isSelected,      // is the cell selected
		   boolean cellHasFocus)    // does the cell have focus
		{
		    @SuppressWarnings("unchecked")
		    JLabel ret = new JLabel( convertToString((T)value, index) );
		    if (isSelected) {
			ret.setBackground(list.getSelectionBackground());
			ret.setForeground(list.getSelectionForeground());
		    } else {
			ret.setBackground(list.getBackground());
			ret.setForeground(list.getForeground());
		    }
		    ret.setEnabled(list.isEnabled());
		    ret.setFont(list.getFont());
		    ret.setOpaque(true);
		    return ret;
		 }
	    });
	}
	
	public abstract String convertToString(T val, int index);

	/** add an element to the end of the list */
	public void addElement( T elem ) {
	   ListModel lm = this.getModel();
	   if (!( lm instanceof DefaultListModel ))
	       throw new RuntimeException("Wrong listmodel for this opeartion to work");
	    DefaultListModel dlm = (DefaultListModel)lm;
	    dlm.addElement( elem );
	}
	

    }

    /** Container */
    public static class Container<T> {
	private T val;
	/** Construct a new container */
	public Container(T i) {
	    val=i;
	}
	/** set the container to a value */
	public void set(T i ) {
	    val = i;
	}
	/** get hte container value */
	public T get() {
	    return val;
	}
    }



}
