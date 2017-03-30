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
package org.fairsim.controller;

import java.awt.event.KeyEvent;
import java.util.zip.DataFormatException;
import org.fairsim.livemode.SimSequenceExtractor;

/**
 * gui for sync options
 * @author m.lachetta
 */
public class SyncPanel extends javax.swing.JPanel implements EasyGui.Sync{
    SimSequenceExtractor seqDetection;
    /**
     * Creates new form SyncPanel
     */
    public SyncPanel() {
        initComponents();
    }
    
    /**
     * enables this panel
     * @param seqDetection sequence extractor 
     */
    void enablePanel(SimSequenceExtractor seqDetection) {
        this.seqDetection = seqDetection;
        syncDelayLabel.setText("Delay: " + seqDetection.getSyncDelay());
        syncDelayTextField.setText(Integer.toString(seqDetection.getSyncDelay()));
        syncAvrLabel.setText("Average: " + seqDetection.getSyncAvr());
        syncAvrTextField.setText(Integer.toString(seqDetection.getSyncAvr()));
        syncFreqLabel.setText("Frequency: " + seqDetection.getSyncFreq());
        syncFreqTextField.setText(Integer.toString(seqDetection.getSyncFreq()));
    }
    
    /**
     * sets the sync delay
     */
    private void setDelay() {
        try {
            seqDetection.setSyncDelay(Integer.parseInt(syncDelayTextField.getText()));
        } catch (NumberFormatException | DataFormatException e) {
        }
        syncDelayLabel.setText("Delay: " + seqDetection.getSyncDelay());
    }
    
    /**
     * sets the sync average
     */
    private void setAvr() {
        try {
            seqDetection.setSyncAvr(Integer.parseInt(syncAvrTextField.getText()));
        } catch (NumberFormatException | DataFormatException e) {
        }
        syncAvrLabel.setText("Average: " + seqDetection.getSyncAvr());
    }
    
    /**
     * sets the sync frequency
     */
    private void setFreq() {
        try {
            seqDetection.setSyncFreq(Integer.parseInt(syncFreqTextField.getText()));
        } catch (NumberFormatException | DataFormatException e) {
        }
        syncFreqLabel.setText("Frequency: " + seqDetection.getSyncFreq());
    }
    
    @Override
    public void setRo(EasyGui.RunningOrder ro) {
        syncDelayTextField.setText(String.valueOf(ro.syncDelay));
        syncFreqTextField.setText(String.valueOf(ro.syncFreq));
        setDelay();
        setFreq();
        return;
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        syncDelayLabel = new javax.swing.JLabel();
        syncAvrLabel = new javax.swing.JLabel();
        syncFreqLabel = new javax.swing.JLabel();
        syncFreqButton = new javax.swing.JButton();
        syncAvrButton = new javax.swing.JButton();
        syncDelayButton = new javax.swing.JButton();
        syncDelayTextField = new javax.swing.JTextField();
        syncAvrTextField = new javax.swing.JTextField();
        syncFreqTextField = new javax.swing.JTextField();

        setBorder(javax.swing.BorderFactory.createTitledBorder("Sync"));

        syncDelayLabel.setText("Delay: -----");

        syncAvrLabel.setText("Average: -----");

        syncFreqLabel.setText("Frequency: -----");

        syncFreqButton.setText("Set Frequency");
        syncFreqButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                syncFreqButtonActionPerformed(evt);
            }
        });

        syncAvrButton.setText("Set Average");
        syncAvrButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                syncAvrButtonActionPerformed(evt);
            }
        });

        syncDelayButton.setText("Set Delay");
        syncDelayButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                syncDelayButtonActionPerformed(evt);
            }
        });

        syncDelayTextField.setText("0");
        syncDelayTextField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                syncDelayTextFieldKeyPressed(evt);
            }
        });

        syncAvrTextField.setText("0");
        syncAvrTextField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                syncAvrTextFieldKeyPressed(evt);
            }
        });

        syncFreqTextField.setText("0");
        syncFreqTextField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                syncFreqTextFieldKeyPressed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(syncFreqLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(syncDelayLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(syncAvrLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(syncDelayButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(syncAvrButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(syncFreqButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(syncDelayTextField)
                    .addComponent(syncAvrTextField)
                    .addComponent(syncFreqTextField, javax.swing.GroupLayout.DEFAULT_SIZE, 50, Short.MAX_VALUE))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(syncDelayButton)
                    .addComponent(syncDelayTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(syncDelayLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(syncAvrButton)
                    .addComponent(syncAvrTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(syncAvrLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(syncFreqLabel)
                    .addComponent(syncFreqButton)
                    .addComponent(syncFreqTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void syncFreqButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_syncFreqButtonActionPerformed
        setFreq();
    }//GEN-LAST:event_syncFreqButtonActionPerformed

    private void syncAvrButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_syncAvrButtonActionPerformed
        setAvr();
    }//GEN-LAST:event_syncAvrButtonActionPerformed

    private void syncDelayButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_syncDelayButtonActionPerformed
        setDelay();
    }//GEN-LAST:event_syncDelayButtonActionPerformed

    private void syncDelayTextFieldKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_syncDelayTextFieldKeyPressed
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            setDelay();
        }
    }//GEN-LAST:event_syncDelayTextFieldKeyPressed

    private void syncAvrTextFieldKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_syncAvrTextFieldKeyPressed
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            setAvr();
        }
    }//GEN-LAST:event_syncAvrTextFieldKeyPressed

    private void syncFreqTextFieldKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_syncFreqTextFieldKeyPressed
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            setFreq();
        }
    }//GEN-LAST:event_syncFreqTextFieldKeyPressed


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton syncAvrButton;
    private javax.swing.JLabel syncAvrLabel;
    private javax.swing.JTextField syncAvrTextField;
    private javax.swing.JButton syncDelayButton;
    private javax.swing.JLabel syncDelayLabel;
    private javax.swing.JTextField syncDelayTextField;
    private javax.swing.JButton syncFreqButton;
    private javax.swing.JLabel syncFreqLabel;
    private javax.swing.JTextField syncFreqTextField;
    // End of variables declaration//GEN-END:variables
}
