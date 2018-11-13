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

import java.util.ArrayList;
import java.util.List;
import org.fairsim.livemode.LiveControlPanel;
import org.fairsim.utils.Conf;
import org.fairsim.utils.Tool;

/**
 * class for the advanced controller gui
 *
 * @author m.lachetta
 */
public class AdvancedGui extends javax.swing.JPanel implements EasyGui.AdvGui, MotherGui{

    LiveControlPanel motherGui;
    private String controllerAdress;
    private final String[] camAdresses;
    private int camCounts;
    private static final int camPort = 32323, controllerPort = 32322;
    private static final int CAMCOUNTMAX = 3;

    /**
     * Creates a new advanced controller gui
     *
     * @param cfg Configuration settings
     * @param channelNames active channels of fairSIM
     * @param motherGui the main gui of fairSIM
     */
    public AdvancedGui(Conf.Folder cfg, String[] channelNames, LiveControlPanel motherGui) {
        initComponents();
        this.motherGui = motherGui;
        // set number of cameras
        camCounts = channelNames.length;
        if (camCounts > CAMCOUNTMAX) {
            camCounts = CAMCOUNTMAX;
        }
        camAdresses = new String[camCounts];
        // readin controller adress
        try {
            controllerAdress = cfg.getStr("ControllerAdress").val();
        } catch (Conf.EntryNotFoundException ex) {
            controllerAdress = "localhost";
            Tool.error("[fairSIM] No ControllerAdress found. ControllerAdress set to 'localhost'", false);
        }
        // readin camera adresses
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
                for (int j = 0; j < camCounts; j++) {
                    if (i != j && camAdresses[i].equals(camAdresses[j])) {
                        camAdresses[j] = null;
                        Tool.trace("[fairSIM] Camera adress of channel '" + channelNames[j] + "' equals camera adress of channel " + channelNames[i]);
                    }
                }
            }
        }
        // init controller panel
        controllerPanel.enablePanel(this, controllerAdress, controllerPort, motherGui.getSequenceExtractor());
        serverLabel.setText("Controller: " + controllerAdress);
        // init camera panels
        if (camCounts > 0 && camAdresses[0] != null) {
            camControllerPanel0.enablePanel(this, camAdresses[0], camPort, channelNames[0]);
            serverLabel.setText(serverLabel.getText() + "   Camera_0: " + camAdresses[0]);
        } else {
            camControllerPanel0.disablePanel();
        }
        if (camCounts > 1 && camAdresses[1] != null) {
            camControllerPanel1.enablePanel(this, camAdresses[1], camPort, channelNames[1]);
            serverLabel.setText(serverLabel.getText() + "   Camera_1: " + camAdresses[1]);
        } else {
            camControllerPanel1.disablePanel();
        }
        if (camCounts > 2 && camAdresses[2] != null) {
            camControllerPanel2.enablePanel(this, camAdresses[2], camPort, channelNames[2]);
            serverLabel.setText(serverLabel.getText() + "   Camera_2: " + camAdresses[2]);
        } else {
            camControllerPanel2.disablePanel();
        }
        // init sync & registration panel
        syncPanel.enablePanel(motherGui.getSequenceExtractor());
        registrationPanel.enablePanel(cfg, channelNames, motherGui.getSequenceExtractor(), motherGui.getReconRunner());
    }

    /**
     * Shows a new text-line in the log
     *
     * @param text String that should be displayed in the log
     */
    @Override
    public void showText(String text) {
        logger.append(text + "\n");
    }

    @Override
    public EasyGui.Ctrl getCtrl() {
        return this.controllerPanel;
    }

    @Override
    public EasyGui.Sync getSync() {
        return this.syncPanel;
    }

    @Override
    public EasyGui.Reg getReg() {
        return this.registrationPanel;
    }

    @Override
    public List<EasyGui.Cam> getCams() {
        List<EasyGui.Cam> camGuis = new ArrayList<>();
        if (camControllerPanel0.enabled) {
            camGuis.add(camControllerPanel0);
        }
        if (camControllerPanel1.enabled) {
            camGuis.add(camControllerPanel1);
        }
        if (camControllerPanel2.enabled) {
            camGuis.add(camControllerPanel2);
        }
        return camGuis;
    }

    @Override
    public void setRo(EasyGui.RunningOrder ro) throws EasyGui.EasyGuiException {
        int size = ro.allowBigRoi ? 512 : 256;
        if (motherGui.getWfSize() == size) {
            return;
        }

        if (size == 512) {
            refreshBox.setSelectedIndex(0);
        } else if (size == 256) {
            refreshBox.setSelectedIndex(1);
        } else {
            throw new EasyGui.EasyGuiException("AdvancedGui: no ROI found");
        }
        refreshView();
    }

    /**
     * refreshes the {@link org.fairsim.sim_gui.PlainImageDisplay} for widefield
     * reconstructed image with the selected image size from this gui
     */
    private void refreshView() {
        int ps = Integer.parseInt(refreshBox.getSelectedItem().toString());
        if (ps > 0) {
            refreshButton.setEnabled(false);
            motherGui.refreshView(ps);
            refreshButton.setEnabled(true);
        }
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
        logger = new java.awt.TextArea();
        refreshButton = new javax.swing.JButton();
        refreshBox = new javax.swing.JComboBox<>();
        camControllerPanel0 = new org.fairsim.controller.CameraPanel();
        camControllerPanel1 = new org.fairsim.controller.CameraPanel();
        camControllerPanel2 = new org.fairsim.controller.CameraPanel();
        controllerPanel = new org.fairsim.controller.ControllerPanel();
        syncPanel = new org.fairsim.controller.SyncPanel();
        registrationPanel = new org.fairsim.controller.RegistrationPanel();

        clientServerPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Client-Server-Communication"));

        serverLabel.setFont(new java.awt.Font("Tahoma", 0, 12)); // NOI18N
        serverLabel.setText("Controller: -   Camera_1: -   Camera_2: -    Camera_3: -");

        logger.setEditable(false);

        refreshButton.setText("Refresh View");
        refreshButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                refreshButtonActionPerformed(evt);
            }
        });

        refreshBox.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "512", "256" }));

        javax.swing.GroupLayout clientServerPanelLayout = new javax.swing.GroupLayout(clientServerPanel);
        clientServerPanel.setLayout(clientServerPanelLayout);
        clientServerPanelLayout.setHorizontalGroup(
            clientServerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(clientServerPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(clientServerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(clientServerPanelLayout.createSequentialGroup()
                        .addComponent(serverLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(refreshBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(refreshButton))
                    .addComponent(logger, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        clientServerPanelLayout.setVerticalGroup(
            clientServerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(clientServerPanelLayout.createSequentialGroup()
                .addGroup(clientServerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(serverLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(clientServerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                        .addComponent(refreshButton)
                        .addComponent(refreshBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(logger, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(camControllerPanel0, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(camControllerPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(camControllerPanel2, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(controllerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(syncPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(registrationPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                    .addComponent(clientServerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(controllerPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(syncPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(registrationPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(camControllerPanel0, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(camControllerPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(camControllerPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(clientServerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    private void refreshButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_refreshButtonActionPerformed
        refreshView();
    }//GEN-LAST:event_refreshButtonActionPerformed


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private org.fairsim.controller.CameraPanel camControllerPanel0;
    private org.fairsim.controller.CameraPanel camControllerPanel1;
    private org.fairsim.controller.CameraPanel camControllerPanel2;
    private javax.swing.JPanel clientServerPanel;
    private org.fairsim.controller.ControllerPanel controllerPanel;
    private java.awt.TextArea logger;
    javax.swing.JComboBox<String> refreshBox;
    private javax.swing.JButton refreshButton;
    private org.fairsim.controller.RegistrationPanel registrationPanel;
    private javax.swing.JLabel serverLabel;
    private org.fairsim.controller.SyncPanel syncPanel;
    // End of variables declaration//GEN-END:variables
}
