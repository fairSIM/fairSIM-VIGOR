/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fairsim.sim_gui;

import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import org.fairsim.livemode.ReconstructionRunner;

/**
 *
 * @author Mario
 */
public class RawImageDisplay extends PlainImageDisplay {
    
    public RawImageDisplay(ReconstructionRunner recRunner, String... names) {
        super(recRunner.nrChannels, recRunner.width, recRunner.height, names);
        JLabel rawLabel = new JLabel("Raw Frame: " + recRunner.rawOutput);
        JSlider rawSlider = new JSlider(JSlider.HORIZONTAL, -1, recRunner.nrDirs * recRunner.nrPhases - 1, -1);
        rawSlider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                recRunner.rawOutput = rawSlider.getValue();
                rawLabel.setText("Raw Frame: " + recRunner.rawOutput);
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
}
