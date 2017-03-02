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

import org.fairsim.livemode.ReconstructionRunner;
import org.fairsim.livemode.SimSequenceExtractor;
import org.fairsim.utils.Conf;
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class ControllerGui extends javax.swing.JPanel {
    private static final int CAMCOUNTMAX = 3;
    private String controllerAdress;
    private int port, camCounts;
    private final String[] camAdresses;

    /**
     * Creates the GUI for the Controller
     *
     * @param cfg Configuration settings
     * @param channelNames Camera Channels
     * @param seqDetection The Sim-Sequence-Extractor
     */
    public ControllerGui(Conf.Folder cfg, String[] channelNames, SimSequenceExtractor seqDetection, ReconstructionRunner recRunner) {
        initComponents();
        
        camCounts = channelNames.length;
        if (camCounts > 3) camCounts = CAMCOUNTMAX;
        
        //readin port
        try {
            port = cfg.getInt("TCPPort").val();
        } catch (Conf.EntryNotFoundException ex) {
            port = 32322;
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
        //init controller panels
        controllerPanel.enablePanel(this, controllerAdress, port, seqDetection);
        serverLabel.setText("Controller: " + controllerAdress);
        
        if (camCounts > 0 && camAdresses[0] != null) {
            camControllerPanel0.enablePanel(this, camAdresses[0], port, channelNames[0]);
            serverLabel.setText(serverLabel.getText() + "   Camera_0: " + camAdresses[0]);
        }
        else camControllerPanel0.disablePanel();
        if (camCounts > 1 && camAdresses[1] != null) {
            camControllerPanel1.enablePanel(this, camAdresses[1], port, channelNames[1]);
            serverLabel.setText(serverLabel.getText() + "   Camera_1: " + camAdresses[1]);
        }
        else camControllerPanel1.disablePanel();
        if (camCounts > 2 && camAdresses[2] != null) {
            camControllerPanel2.enablePanel(this, camAdresses[2], port, channelNames[2]);
            serverLabel.setText(serverLabel.getText() + "   Camera_2: " + camAdresses[2]);
        }
        else camControllerPanel2.disablePanel();
        
        syncPanel.enablePanel(seqDetection);
        registrationPanel.enablePanel(cfg, channelNames, seqDetection, recRunner);
    }
    
    /**
     * Shows a new text-line in the text-field
     *
     * @param text String that shoulds be displayed in the text-field
     */
    public void showText(String text) {
        logger.append(text + "\n");
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
        camControllerPanel0 = new org.fairsim.controller.CameraPanel();
        camControllerPanel1 = new org.fairsim.controller.CameraPanel();
        camControllerPanel2 = new org.fairsim.controller.CameraPanel();
        controllerPanel = new org.fairsim.controller.ControllerPanel();
        syncPanel = new org.fairsim.controller.SyncPanel();
        registrationPanel = new org.fairsim.controller.RegistrationPanel();

        clientServerPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Client-Server-Communication"));

        serverLabel.setFont(new java.awt.Font("Tahoma", 0, 12)); // NOI18N
        serverLabel.setText("Controller: -   Camera_1: -   Camera_2: -    Camera_3: -");

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
                    .addComponent(logger, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        clientServerPanelLayout.setVerticalGroup(
            clientServerPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(clientServerPanelLayout.createSequentialGroup()
                .addComponent(serverLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
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
                    .addComponent(clientServerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addComponent(camControllerPanel2, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(camControllerPanel1, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(camControllerPanel0, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(controllerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addContainerGap())
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addComponent(syncPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(registrationPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(0, 0, Short.MAX_VALUE))))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(controllerPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(camControllerPanel0, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(camControllerPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(camControllerPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(syncPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(registrationPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(clientServerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    
    
    

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private org.fairsim.controller.CameraPanel camControllerPanel0;
    private org.fairsim.controller.CameraPanel camControllerPanel1;
    private org.fairsim.controller.CameraPanel camControllerPanel2;
    private javax.swing.JPanel clientServerPanel;
    private org.fairsim.controller.ControllerPanel controllerPanel;
    private java.awt.TextArea logger;
    private org.fairsim.controller.RegistrationPanel registrationPanel;
    private javax.swing.JLabel serverLabel;
    private org.fairsim.controller.SyncPanel syncPanel;
    // End of variables declaration//GEN-END:variables
}
