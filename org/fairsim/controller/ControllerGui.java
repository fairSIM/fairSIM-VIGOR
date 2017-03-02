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
import java.io.FileNotFoundException;
import java.util.zip.DataFormatException;
import org.fairsim.livemode.ReconstructionRunner;
import org.fairsim.livemode.SimSequenceExtractor;
import org.fairsim.registration.RegFileCreatorGui;
import org.fairsim.registration.Registration;
import org.fairsim.utils.Conf;
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class ControllerGui extends javax.swing.JPanel {

    
    private static final int CAMCOUNTMAX = 3;
    
    
    String controllerAdress, regFolder;
    int tcpPort, camCounts;
    String[] channelNames, camAdresses;
    SimSequenceExtractor seqDetection;
    ReconstructionRunner recRunner;

    /**
     * Creates the GUI for the Controller
     *
     * @param cfg Configuration settings
     * @param channelNames Camera Channels
     * @param seqDetection The Sim-Sequence-Extractor
     */
    public ControllerGui(Conf.Folder cfg, String[] channelNames, SimSequenceExtractor seqDetection, ReconstructionRunner recRunner) {
        this.seqDetection = seqDetection;
        this.recRunner = recRunner;
        this.channelNames = channelNames;
        camCounts = channelNames.length;
        if (camCounts > 3) camCounts = CAMCOUNTMAX;
        readinFromXML(cfg);
        
        initComponents();
        if (camCounts > 0) camControllerPanel0.enablePanel(this, camAdresses[0], tcpPort, channelNames[0]);
        else camControllerPanel0.disablePanel();
        if (camCounts > 1) camControllerPanel1.enablePanel(this, camAdresses[1], tcpPort, channelNames[1]);
        else camControllerPanel1.disablePanel();
        if (camCounts > 2) camControllerPanel2.enablePanel(this, camAdresses[2], tcpPort, channelNames[2]);
        else camControllerPanel2.disablePanel();

        initSync();
        initReg(cfg);
    }
    
    private void readinFromXML(Conf.Folder cfg) {
        //readin port
        try {
            tcpPort = cfg.getInt("TCPPort").val();
        } catch (Conf.EntryNotFoundException ex) {
            tcpPort = 32322;
            Tool.error("[fairSIM] No TCPPort found. TCPPort set to '32322'", false);
        }
        //readin controller adress
        try {
            controllerAdress = cfg.getStr("ControllerAdress").val();
        } catch (Conf.EntryNotFoundException ex) {
            controllerAdress = "localhost";
            Tool.error("[fairSIM] No ControllerAdress found. ControllerAdress set to 'localhost'", false);
        }
        //readin cam adresses
        camAdresses = new String[camCounts];
        for (int i = 0; i < camCounts; i++) {
            try {
                Conf.Folder fld = cfg.cd("channel-" + channelNames[i]);
                camAdresses[i] = fld.getStr("CamAdress").val();
            } catch (Conf.EntryNotFoundException ex) {
                camAdresses[i] = null;
                Tool.error("[fairSIM] No camera adress found for channel " + channelNames[i], false);
            }
        }
        //check for doublicates
        for (int i = 0; i < camCounts; i++) {
            if (camAdresses[i] != null) {
                if (camAdresses[i].equals(controllerAdress)) {
                    camAdresses[i] = null;
                    Tool.error("[fairSIM] Camera adress of channel '" + channelNames[i] + "' equals controller adress", false);
                } else {
                    for (int j = 0; j < camCounts; j++) {
                        if (i != j && camAdresses[i].equals(camAdresses[j])) {
                            camAdresses[j] = null;
                            Tool.error("[fairSIM] Camera adress of channel '" + channelNames[j] + "' equals camera adress of channel " + channelNames[i], false);
                        }
                    }
                }
            }
        }
    }

    

    /**
     * Initialises the GUI for the syncronisation
     */
    private void initSync() {
        syncDelayLabel.setText("Delay: " + seqDetection.getSyncDelay());
        syncDelayTextField.setText(Integer.toString(seqDetection.getSyncDelay()));
        syncAvrLabel.setText("Average: " + seqDetection.getSyncAvr());
        syncAvrTextField.setText(Integer.toString(seqDetection.getSyncAvr()));
        syncFreqLabel.setText("Frequency: " + seqDetection.getSyncFreq());
        syncFreqTextField.setText(Integer.toString(seqDetection.getSyncFreq()));
    }

    /**
     * Initialises the GUI for the Registration
     *
     * @param cfg Configuration
     * @param channels Camera channels
     */
    private void initReg(Conf.Folder cfg) {
        try {
            regFolder = Registration.getRegFolder(cfg);
            Class.forName("ij.ImagePlus");
            Class.forName("ij.process.ImageProcessor");
            Class.forName("bunwarpj.Transformation");
            Class.forName("bunwarpj.bUnwarpJ_");
        } catch (FileNotFoundException ex) {
            regPanel.setEnabled(false);
            regReconButton.setEnabled(false);
            regWfButton.setEnabled(false);
            regCreatorButton.setEnabled(false);
            Tool.error("[fairSIM] " + ex.getMessage(), false);
        } catch (ClassNotFoundException ex) {
            regCreatorButton.setEnabled(false);
            Tool.error("[fairSIM] Jar files for bunwarpj and/or imagej are missing. Deaktived registration-creator.", false);
        }
        for (final String channel : channelNames) {
            new Thread(new Runnable() {
                @Override
                public void run() {
                    Registration.createRegistration(regFolder, channel);
                }
            }).start();
        }
    }

    /**
     * Shows a new text-line in the text-field
     *
     * @param text String that shoulds be displayed in the text-field
     */
    public void showText(String text) {
        textArea1.append(text + "\n");
    }

    /**
     * Sets server label
     *
     * @param adress server adress
     * @param port server port
     */
    private void setConnectionLabel(String adress, int port) {
        serverLabel.setText("Server: " + adress + ":" + port);
    }

    

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        clientServerPanel = new javax.swing.JPanel();
        serverLabel = new javax.swing.JLabel();
        textArea1 = new java.awt.TextArea();
        softwarePanel = new javax.swing.JPanel();
        syncDelayButton = new javax.swing.JButton();
        syncDelayTextField = new javax.swing.JTextField();
        syncAvrButton = new javax.swing.JButton();
        syncAvrTextField = new javax.swing.JTextField();
        syncDelayLabel = new javax.swing.JLabel();
        syncAvrLabel = new javax.swing.JLabel();
        syncFreqLabel = new javax.swing.JLabel();
        syncFreqButton = new javax.swing.JButton();
        syncFreqTextField = new javax.swing.JTextField();
        regPanel = new javax.swing.JPanel();
        regWfButton = new javax.swing.JToggleButton();
        regReconButton = new javax.swing.JToggleButton();
        regCreatorButton = new javax.swing.JButton();
        camControllerPanel0 = new org.fairsim.controller.CameraPanel();
        camControllerPanel1 = new org.fairsim.controller.CameraPanel();
        camControllerPanel2 = new org.fairsim.controller.CameraPanel();
        controllerPanel1 = new org.fairsim.controller.ControllerPanel();

        clientServerPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Client-Server-Communication"));

        serverLabel.setFont(new java.awt.Font("Tahoma", 1, 14)); // NOI18N
        serverLabel.setText("Server: -");

        javax.swing.GroupLayout clientServerPanelLayout = new javax.swing.GroupLayout(clientServerPanel);
        clientServerPanel.setLayout(clientServerPanelLayout);
        clientServerPanelLayout.setHorizontalGroup(
            clientServerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(clientServerPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(clientServerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(clientServerPanelLayout.createSequentialGroup()
                        .addComponent(serverLabel)
                        .addGap(0, 0, Short.MAX_VALUE))
                    .addComponent(textArea1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        clientServerPanelLayout.setVerticalGroup(
            clientServerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(clientServerPanelLayout.createSequentialGroup()
                .addComponent(serverLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(textArea1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        softwarePanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Sync-Controller"));

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

        syncAvrButton.setText("Set Average");
        syncAvrButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                syncAvrButtonActionPerformed(evt);
            }
        });

        syncAvrTextField.setText("0");

        syncDelayLabel.setText("Delay: -----");

        syncAvrLabel.setText("Average: -----");

        syncFreqLabel.setText("Frequency: -----");

        syncFreqButton.setText("Set Frequency");
        syncFreqButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                syncFreqButtonActionPerformed(evt);
            }
        });

        syncFreqTextField.setText("0");

        javax.swing.GroupLayout softwarePanelLayout = new javax.swing.GroupLayout(softwarePanel);
        softwarePanel.setLayout(softwarePanelLayout);
        softwarePanelLayout.setHorizontalGroup(
            softwarePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(softwarePanelLayout.createSequentialGroup()
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addGroup(softwarePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(syncFreqLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(syncDelayLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(syncAvrLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(softwarePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(syncDelayButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(syncAvrButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(syncFreqButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(softwarePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(syncDelayTextField, javax.swing.GroupLayout.DEFAULT_SIZE, 50, Short.MAX_VALUE)
                    .addComponent(syncAvrTextField)
                    .addComponent(syncFreqTextField)))
        );
        softwarePanelLayout.setVerticalGroup(
            softwarePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(softwarePanelLayout.createSequentialGroup()
                .addGroup(softwarePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(syncDelayButton)
                    .addComponent(syncDelayTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(syncDelayLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(softwarePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(syncAvrButton)
                    .addComponent(syncAvrTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(syncAvrLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(softwarePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(syncFreqLabel)
                    .addComponent(syncFreqButton)
                    .addComponent(syncFreqTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
        );

        regPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Image Registration"));

        regWfButton.setText("Register In Widefield");
        regWfButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                regWfButtonActionPerformed(evt);
            }
        });

        regReconButton.setText("Register In Reconstruction");
        regReconButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                regReconButtonActionPerformed(evt);
            }
        });

        regCreatorButton.setText("Create Registration File");
        regCreatorButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                regCreatorButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout regPanelLayout = new javax.swing.GroupLayout(regPanel);
        regPanel.setLayout(regPanelLayout);
        regPanelLayout.setHorizontalGroup(
            regPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(regPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(regPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(regWfButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(regReconButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(regCreatorButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        regPanelLayout.setVerticalGroup(
            regPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, regPanelLayout.createSequentialGroup()
                .addComponent(regWfButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(regReconButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(regCreatorButton))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(clientServerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(softwarePanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(regPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addComponent(camControllerPanel2, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(camControllerPanel1, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(camControllerPanel0, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(controllerPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addContainerGap())))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(controllerPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(camControllerPanel0, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(camControllerPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(camControllerPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(softwarePanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(regPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(clientServerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    private void syncDelayButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_syncDelayButtonActionPerformed
        setSyncDelay();
    }//GEN-LAST:event_syncDelayButtonActionPerformed

    private void syncAvrButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_syncAvrButtonActionPerformed
        try {
            seqDetection.setSyncAvr(Integer.parseInt(syncAvrTextField.getText()));
        } catch (NumberFormatException e) {
        } catch (DataFormatException ex) {
        }
        syncAvrLabel.setText("Average: " + seqDetection.getSyncAvr());
    }//GEN-LAST:event_syncAvrButtonActionPerformed

    private void syncFreqButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_syncFreqButtonActionPerformed
        try {
            seqDetection.setSyncFreq(Integer.parseInt(syncFreqTextField.getText()));
        } catch (NumberFormatException e) {
        } catch (DataFormatException ex) {
        }
        syncFreqLabel.setText("Frequency: " + seqDetection.getSyncFreq());
    }//GEN-LAST:event_syncFreqButtonActionPerformed

    private void regWfButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_regWfButtonActionPerformed
        Registration.setWidefield(regWfButton.isSelected());
    }//GEN-LAST:event_regWfButtonActionPerformed

    private void regReconButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_regReconButtonActionPerformed
        Registration.setRecon(regReconButton.isSelected());
    }//GEN-LAST:event_regReconButtonActionPerformed

    private void regCreatorButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_regCreatorButtonActionPerformed
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new RegFileCreatorGui(regFolder, channelNames, seqDetection, recRunner, regWfButton, regReconButton).setVisible(true);
            }
        });
    }//GEN-LAST:event_regCreatorButtonActionPerformed

    private void syncDelayTextFieldKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_syncDelayTextFieldKeyPressed
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            setSyncDelay();
        }
    }//GEN-LAST:event_syncDelayTextFieldKeyPressed

    void setSyncDelay() {
        try {
            seqDetection.setSyncDelay(Integer.parseInt(syncDelayTextField.getText()));
        } catch (NumberFormatException e) {
        } catch (DataFormatException ex) {
        }
        syncDelayLabel.setText("Delay: " + seqDetection.getSyncDelay());
    }
    
    

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private org.fairsim.controller.CameraPanel camControllerPanel0;
    private org.fairsim.controller.CameraPanel camControllerPanel1;
    private org.fairsim.controller.CameraPanel camControllerPanel2;
    private javax.swing.JPanel clientServerPanel;
    private org.fairsim.controller.ControllerPanel controllerPanel1;
    private javax.swing.JButton regCreatorButton;
    private javax.swing.JPanel regPanel;
    public javax.swing.JToggleButton regReconButton;
    public javax.swing.JToggleButton regWfButton;
    private javax.swing.JLabel serverLabel;
    private javax.swing.JPanel softwarePanel;
    private javax.swing.JButton syncAvrButton;
    private javax.swing.JLabel syncAvrLabel;
    private javax.swing.JTextField syncAvrTextField;
    private javax.swing.JButton syncDelayButton;
    private javax.swing.JLabel syncDelayLabel;
    private javax.swing.JTextField syncDelayTextField;
    private javax.swing.JButton syncFreqButton;
    private javax.swing.JLabel syncFreqLabel;
    private javax.swing.JTextField syncFreqTextField;
    private java.awt.TextArea textArea1;
    // End of variables declaration//GEN-END:variables
}
