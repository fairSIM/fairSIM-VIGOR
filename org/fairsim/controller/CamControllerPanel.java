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

/**
 *
 * @author m.lachetta
 */
public class CamControllerPanel extends javax.swing.JPanel {
    private ControllerGui cg;
    int camId;
    
    /**
     * Creates new form CamControllerPanel
     */
    public CamControllerPanel() {
        initComponents();
    }
    
    public void init(ControllerGui cg, int camId) {
        this.cg = cg;
        this.camId = camId;
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        cam0ChannelLabel = new javax.swing.JLabel();
        cam0RoiLabel = new javax.swing.JLabel();
        cam0ExposureLabel = new javax.swing.JLabel();
        cam0StartButton = new javax.swing.JButton();
        cam0StopButton = new javax.swing.JButton();
        cam0FpsLabel = new javax.swing.JLabel();
        cam0QueuingPanel = new javax.swing.JPanel();
        cam0QueuingLabel = new javax.swing.JLabel();
        cam0SendingPanel = new javax.swing.JPanel();
        cam0SendingLabel = new javax.swing.JLabel();
        cam0RoiXField = new javax.swing.JTextField();
        cam0RoiYField = new javax.swing.JTextField();
        cam0RoiWField = new javax.swing.JTextField();
        cam0RoiHField = new javax.swing.JTextField();
        cam0RoiButton = new javax.swing.JButton();
        cam0ExposureField = new javax.swing.JTextField();
        cam0MsLabel = new javax.swing.JLabel();
        cam0ExposureButton = new javax.swing.JButton();
        cam0GroupBox = new javax.swing.JComboBox<>();
        cam0ConfigBox = new javax.swing.JComboBox<>();
        cam0ConfigButton = new javax.swing.JButton();

        setBorder(javax.swing.BorderFactory.createTitledBorder("Camera-Controller"));

        cam0ChannelLabel.setText("Channel: - ");

        cam0RoiLabel.setText("ROI: -");

        cam0ExposureLabel.setText("Exposure Time: - ");

        cam0StartButton.setText("Start Acquisition");
        cam0StartButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam0StartButtonActionPerformed(evt);
            }
        });

        cam0StopButton.setText("Stop Acquisition");
        cam0StopButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam0StopButtonActionPerformed(evt);
            }
        });

        cam0FpsLabel.setText("FPS: -");

        cam0QueuingPanel.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        cam0QueuingLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        cam0QueuingLabel.setText("Image Queuing");

        javax.swing.GroupLayout cam0QueuingPanelLayout = new javax.swing.GroupLayout(cam0QueuingPanel);
        cam0QueuingPanel.setLayout(cam0QueuingPanelLayout);
        cam0QueuingPanelLayout.setHorizontalGroup(
            cam0QueuingPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(cam0QueuingPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(cam0QueuingLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        cam0QueuingPanelLayout.setVerticalGroup(
            cam0QueuingPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(cam0QueuingLabel)
        );

        cam0SendingPanel.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        cam0SendingLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        cam0SendingLabel.setText("Image Sending");

        javax.swing.GroupLayout cam0SendingPanelLayout = new javax.swing.GroupLayout(cam0SendingPanel);
        cam0SendingPanel.setLayout(cam0SendingPanelLayout);
        cam0SendingPanelLayout.setHorizontalGroup(
            cam0SendingPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(cam0SendingPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(cam0SendingLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        cam0SendingPanelLayout.setVerticalGroup(
            cam0SendingPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(cam0SendingLabel)
        );

        cam0RoiXField.setText("765");

        cam0RoiYField.setText("765");

        cam0RoiWField.setText("520");

        cam0RoiHField.setText("520");

        cam0RoiButton.setText("Set ROI");
        cam0RoiButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam0RoiButtonActionPerformed(evt);
            }
        });

        cam0ExposureField.setText("3.509");

        cam0MsLabel.setText("ms");

        cam0ExposureButton.setText("Set Exposure");
        cam0ExposureButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam0ExposureButtonActionPerformed(evt);
            }
        });

        cam0GroupBox.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                cam0GroupBoxItemStateChanged(evt);
            }
        });

        cam0ConfigButton.setText("Set Config");
        cam0ConfigButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam0ConfigButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(cam0ChannelLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0RoiLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 165, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0ExposureLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 167, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addComponent(cam0RoiXField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0RoiYField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0RoiWField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0RoiHField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addGap(18, 18, 18)
                                .addComponent(cam0RoiButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0ExposureField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0MsLabel)
                                .addGap(18, 18, 18)
                                .addComponent(cam0ExposureButton)))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(cam0GroupBox, javax.swing.GroupLayout.PREFERRED_SIZE, 120, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0ConfigBox, javax.swing.GroupLayout.PREFERRED_SIZE, 120, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0ConfigButton))
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(cam0StartButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0StopButton)))
                        .addGap(0, 0, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(cam0FpsLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(18, 18, 18)
                        .addComponent(cam0QueuingPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(18, 18, 18)
                        .addComponent(cam0SendingPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cam0ChannelLabel)
                    .addComponent(cam0RoiLabel)
                    .addComponent(cam0StopButton)
                    .addComponent(cam0StartButton)
                    .addComponent(cam0ExposureLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(cam0FpsLabel)
                    .addComponent(cam0QueuingPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0SendingPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cam0RoiButton)
                    .addComponent(cam0RoiXField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0RoiYField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0RoiWField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0RoiHField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0ExposureButton)
                    .addComponent(cam0ExposureField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0MsLabel)
                    .addComponent(cam0GroupBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0ConfigBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0ConfigButton))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void cam0StartButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam0StartButtonActionPerformed
        cg.startCam(camId);
    }//GEN-LAST:event_cam0StartButtonActionPerformed

    private void cam0StopButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam0StopButtonActionPerformed
        cg.stopCam(camId);
    }//GEN-LAST:event_cam0StopButtonActionPerformed

    private void cam0RoiButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam0RoiButtonActionPerformed
        cg.setRoi(camId);
    }//GEN-LAST:event_cam0RoiButtonActionPerformed

    private void cam0ExposureButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam0ExposureButtonActionPerformed
        cg.setExposureTime(camId);
    }//GEN-LAST:event_cam0ExposureButtonActionPerformed

    private void cam0GroupBoxItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_cam0GroupBoxItemStateChanged
        cg.groupBoxSelected(camId);
    }//GEN-LAST:event_cam0GroupBoxItemStateChanged

    private void cam0ConfigButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam0ConfigButtonActionPerformed
        cg.setConfig(camId);
    }//GEN-LAST:event_cam0ConfigButtonActionPerformed


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel cam0ChannelLabel;
    private javax.swing.JComboBox<String> cam0ConfigBox;
    private javax.swing.JButton cam0ConfigButton;
    private javax.swing.JButton cam0ExposureButton;
    private javax.swing.JTextField cam0ExposureField;
    private javax.swing.JLabel cam0ExposureLabel;
    private javax.swing.JLabel cam0FpsLabel;
    private javax.swing.JComboBox<String> cam0GroupBox;
    private javax.swing.JLabel cam0MsLabel;
    private javax.swing.JLabel cam0QueuingLabel;
    private javax.swing.JPanel cam0QueuingPanel;
    private javax.swing.JButton cam0RoiButton;
    private javax.swing.JTextField cam0RoiHField;
    private javax.swing.JLabel cam0RoiLabel;
    private javax.swing.JTextField cam0RoiWField;
    private javax.swing.JTextField cam0RoiXField;
    private javax.swing.JTextField cam0RoiYField;
    private javax.swing.JLabel cam0SendingLabel;
    private javax.swing.JPanel cam0SendingPanel;
    private javax.swing.JButton cam0StartButton;
    private javax.swing.JButton cam0StopButton;
    // End of variables declaration//GEN-END:variables
}
