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
package org.fairsim.registration;

import java.awt.Color;
import java.util.zip.DataFormatException;
import javax.swing.JToggleButton;
import org.fairsim.livemode.ReconstructionRunner;
import org.fairsim.livemode.SimSequenceExtractor;

/**
 *
 * @author m.lachetta
 */
public class RegFileCreatorGui extends javax.swing.JFrame {
    
    final RegFileCreator creator;
    final SimSequenceExtractor seqDetection;
    final ReconstructionRunner recRunner;
    final String[] channelNames;
    final String regFolder;
    final JToggleButton wfButton;
    final JToggleButton reconButton;
    
    /**
     * Creates new form RegFileCreatorGui
     */
    public RegFileCreatorGui(String regFolder, String[] channelNames,
            SimSequenceExtractor seqDetection, ReconstructionRunner recRunner,
            JToggleButton wfButton, JToggleButton reconButton) {
        initComponents();
        this.regFolder = regFolder;
        creator = new RegFileCreator(regFolder, channelNames);
        this.channelNames = channelNames;
        this.seqDetection = seqDetection;
        this.recRunner = recRunner;
        this.wfButton = wfButton;
        this.reconButton = reconButton;
        for(String channelName : channelNames) {
            targetComboBox.addItem(channelName);
            sourceComboBox.addItem(channelName);
        }
        if (sourceComboBox.getItemCount() >= 2) sourceComboBox.setSelectedIndex(1);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        targetLabel = new javax.swing.JLabel();
        sourceLabel = new javax.swing.JLabel();
        targetComboBox = new javax.swing.JComboBox<>();
        sourceComboBox = new javax.swing.JComboBox<>();
        modeLabel = new javax.swing.JLabel();
        modeField = new javax.swing.JTextField();
        subsampleLabel = new javax.swing.JLabel();
        subsampleField = new javax.swing.JTextField();
        iDefLabel = new javax.swing.JLabel();
        iDefField = new javax.swing.JTextField();
        fDefLabel = new javax.swing.JLabel();
        fDefField = new javax.swing.JTextField();
        divergenceLabel = new javax.swing.JLabel();
        divergenceField = new javax.swing.JTextField();
        curlLabel = new javax.swing.JLabel();
        curlField = new javax.swing.JTextField();
        landmarkLabel = new javax.swing.JLabel();
        landmarkField = new javax.swing.JTextField();
        imageLabel = new javax.swing.JLabel();
        imageField = new javax.swing.JTextField();
        consistencyLabel = new javax.swing.JLabel();
        consistencyField = new javax.swing.JTextField();
        thresholdLabel = new javax.swing.JLabel();
        thresholdField = new javax.swing.JTextField();
        modeType = new javax.swing.JLabel();
        subsampleType = new javax.swing.JLabel();
        iDefType = new javax.swing.JLabel();
        fDefType = new javax.swing.JLabel();
        divergenceType = new javax.swing.JLabel();
        curlType = new javax.swing.JLabel();
        landmarkType = new javax.swing.JLabel();
        imageType = new javax.swing.JLabel();
        consistencyType = new javax.swing.JLabel();
        thresholdType = new javax.swing.JLabel();
        createButton = new javax.swing.JButton();
        registerButton = new javax.swing.JButton();
        statusLabel = new javax.swing.JLabel();
        deleteButton = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setTitle("Registration-File-Creator");
        setName("regFileCreatorFrame"); // NOI18N

        targetLabel.setText("Moved Channel");

        sourceLabel.setText("Static channel");

        modeLabel.setText("Registration Mode");

        modeField.setText("2");

        subsampleLabel.setText("Image Subsample Factor");

        subsampleField.setText("0");

        iDefLabel.setText("Initial Deformation");

        iDefField.setText("0");

        fDefLabel.setText("Final Deformation");

        fDefField.setText("2");

        divergenceLabel.setText("Divergence Weight");

        divergenceField.setText("0.0");

        curlLabel.setText("Curl Weight");

        curlField.setText("0.0");

        landmarkLabel.setText("Landmark Weight");

        landmarkField.setText("0.0");

        imageLabel.setText("Image Weight");

        imageField.setText("1.0");

        consistencyLabel.setText("Consistency Weight");

        consistencyField.setText("10.0");

        thresholdLabel.setText("Stop Threshold");

        thresholdField.setText("0.01");

        modeType.setText("(int 0-2)");

        subsampleType.setText("(int 0-7)");

        iDefType.setText("(int 0-3)");

        fDefType.setText("(int 0-4)");

        divergenceType.setText("(double)");

        curlType.setText("(double)");

        landmarkType.setText("(double)");

        imageType.setText("(double)");

        consistencyType.setText("(double)");

        thresholdType.setText("(double)");

        createButton.setText("Create new registration file");
        createButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                createButtonActionPerformed(evt);
            }
        });

        registerButton.setText("Register with new file");
        registerButton.setEnabled(false);

        statusLabel.setFont(new java.awt.Font("Tahoma", 1, 14)); // NOI18N
        statusLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        statusLabel.setText("---------");

        deleteButton.setText("Delete Registration");
        deleteButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                deleteButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addComponent(landmarkLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(landmarkField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addComponent(curlLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(curlField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addComponent(divergenceLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(divergenceField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addComponent(fDefLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(fDefField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                    .addComponent(modeLabel)
                                    .addComponent(targetLabel)
                                    .addComponent(sourceLabel)
                                    .addComponent(subsampleLabel)
                                    .addComponent(iDefLabel))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addComponent(iDefField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(subsampleField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                        .addComponent(modeField)
                                        .addComponent(sourceComboBox, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                        .addComponent(targetComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE))))
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addComponent(consistencyLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(consistencyField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addComponent(imageLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(imageField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                                .addComponent(registerButton, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(createButton, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                    .addComponent(thresholdLabel)
                                    .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                    .addComponent(thresholdField, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE))))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(modeType)
                            .addComponent(subsampleType)
                            .addComponent(iDefType)
                            .addComponent(fDefType)
                            .addComponent(divergenceType)
                            .addComponent(curlType)
                            .addComponent(landmarkType)
                            .addComponent(imageType)
                            .addComponent(consistencyType)
                            .addComponent(thresholdType)
                            .addComponent(deleteButton)))
                    .addComponent(statusLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(targetLabel)
                    .addComponent(targetComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(deleteButton))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(sourceLabel)
                    .addComponent(sourceComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(modeLabel)
                    .addComponent(modeField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(modeType))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(subsampleField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(subsampleLabel)
                    .addComponent(subsampleType))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(iDefField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(iDefLabel)
                    .addComponent(iDefType))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(fDefField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(fDefLabel)
                    .addComponent(fDefType))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(divergenceField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(divergenceLabel)
                    .addComponent(divergenceType))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(curlField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(curlLabel)
                    .addComponent(curlType))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(landmarkField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(landmarkLabel)
                    .addComponent(landmarkType))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(imageField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(imageLabel)
                    .addComponent(imageType))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(consistencyField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(consistencyLabel)
                    .addComponent(consistencyType))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(thresholdField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(thresholdLabel)
                    .addComponent(thresholdType))
                .addGap(18, 18, 18)
                .addComponent(createButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(registerButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(statusLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void setBlackStatus(String text) {
        statusLabel.setText(text);
        statusLabel.setForeground(Color.BLACK);
    }
    
    private void setRedStatus(String text) {
        statusLabel.setText(text);
        statusLabel.setForeground(Color.RED);
    }
    
    private void setBlueStatus(String text) {
        statusLabel.setText(text);
        statusLabel.setForeground(Color.BLUE);
    }
    
    private void setCreatorOptions() throws DataFormatException {
        int mode = Integer.parseInt(modeField.getText());
        int img_subsamp_fact = Integer.parseInt(subsampleField.getText());
        int min_scale_deformation = Integer.parseInt(iDefField.getText());
        int max_scale_deformation = Integer.parseInt(fDefField.getText());
        double divWeight = Double.parseDouble(divergenceField.getText());
        double curlWeight = Double.parseDouble(curlField.getText());
        double landmarkWeight = Double.parseDouble(landmarkField.getText());
        double imageWeight = Double.parseDouble(imageField.getText());
        double consistencyWeight = Double.parseDouble(consistencyField.getText());
        double stopThreshold = Double.parseDouble(thresholdField.getText());
        creator.setOptions(mode, img_subsamp_fact, min_scale_deformation,
                max_scale_deformation, divWeight, curlWeight, landmarkWeight,
                imageWeight, consistencyWeight, stopThreshold);
    }
    
    
    
    private void createButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_createButtonActionPerformed
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    createButton.setEnabled(false);
                    setCreatorOptions();

                    int targetId = targetComboBox.getSelectedIndex();
                    int sourceId = sourceComboBox.getSelectedIndex();
                    setBlackStatus("Creating registration file...");
                    seqDetection.setCreatingRegFile(true);
                    
                    creator.createChannelRegFile(targetId, sourceId, recRunner);
                    setBlackStatus("New file created, load new file...");
                    
                    wfButton.setEnabled(false);
                    reconButton.setEnabled(false);
                    boolean wfTemp = Registration.isWidefield();
                    boolean reconTemp = Registration.isRecon();
                    Registration.setWidefield(false);
                    Registration.setRecon(false);
                    Registration.clearRegistration(channelNames[targetId]);
                    Registration.createRegistration(regFolder, channelNames[targetId]);
                    Registration.setWidefield(wfTemp);
                    Registration.setRecon(reconTemp);
                    wfButton.setEnabled(true);
                    reconButton.setEnabled(true);
                    seqDetection.setCreatingRegFile(false);
                    setBlueStatus("New file loaded");
                    
                } catch (Exception ex) {
                    setRedStatus("Error: " + ex.getMessage());
                } finally {
                    createButton.setEnabled(true);
                }
            }
        }).start();
        
    }//GEN-LAST:event_createButtonActionPerformed

    private void deleteButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_deleteButtonActionPerformed
            new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    String deleteChannelName = channelNames[targetComboBox.getSelectedIndex()];
                    Registration.clearRegistration(deleteChannelName);
                    creator.deleteRegFile(deleteChannelName);
                    setBlueStatus("Registration for '" + deleteChannelName + "' deleted");
                } catch (Exception ex) {
                    setRedStatus("Error: " + ex.getMessage());
                }
            }
        }).start();
    }//GEN-LAST:event_deleteButtonActionPerformed

    /**
     * @param args the command line arguments
     */
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JTextField consistencyField;
    private javax.swing.JLabel consistencyLabel;
    private javax.swing.JLabel consistencyType;
    private javax.swing.JButton createButton;
    private javax.swing.JTextField curlField;
    private javax.swing.JLabel curlLabel;
    private javax.swing.JLabel curlType;
    private javax.swing.JButton deleteButton;
    private javax.swing.JTextField divergenceField;
    private javax.swing.JLabel divergenceLabel;
    private javax.swing.JLabel divergenceType;
    private javax.swing.JTextField fDefField;
    private javax.swing.JLabel fDefLabel;
    private javax.swing.JLabel fDefType;
    private javax.swing.JTextField iDefField;
    private javax.swing.JLabel iDefLabel;
    private javax.swing.JLabel iDefType;
    private javax.swing.JTextField imageField;
    private javax.swing.JLabel imageLabel;
    private javax.swing.JLabel imageType;
    private javax.swing.JTextField landmarkField;
    private javax.swing.JLabel landmarkLabel;
    private javax.swing.JLabel landmarkType;
    private javax.swing.JTextField modeField;
    private javax.swing.JLabel modeLabel;
    private javax.swing.JLabel modeType;
    private javax.swing.JButton registerButton;
    private javax.swing.JComboBox<String> sourceComboBox;
    private javax.swing.JLabel sourceLabel;
    private javax.swing.JLabel statusLabel;
    private javax.swing.JTextField subsampleField;
    private javax.swing.JLabel subsampleLabel;
    private javax.swing.JLabel subsampleType;
    private javax.swing.JComboBox<String> targetComboBox;
    private javax.swing.JLabel targetLabel;
    private javax.swing.JTextField thresholdField;
    private javax.swing.JLabel thresholdLabel;
    private javax.swing.JLabel thresholdType;
    // End of variables declaration//GEN-END:variables
}
