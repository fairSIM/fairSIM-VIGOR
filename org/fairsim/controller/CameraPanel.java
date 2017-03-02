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

import java.awt.Color;
import java.awt.Component;
import java.awt.event.KeyEvent;
import java.util.ArrayList;
import java.util.List;
import javax.swing.*;
import org.fairsim.utils.Tool;

/**
 *
 * @author m.lachetta
 */
public class CameraPanel extends javax.swing.JPanel implements ClientPanel {

    private ControllerGui motherGui;
    private CameraClient client;
    private String channelName;
    boolean instructionDone;
    private Color defaultColor;
    private StatusUpdateThread updateThread;
    private List<JButton> buttons;
    private List<Component> components;
    private static final int ROILENGTH = 4, UPDATEDELAY = 2000;

    /**
     * Creates new form CamControllerPanel
     */
    public CameraPanel() {
        initComponents();
    }

    private void init() {
        defaultColor = queuingPanel.getBackground();

        components = new ArrayList<>();
        components.add(channelLabel);
        components.add(exposureLabel);
        components.add(msLabel);
        components.add(roiLabel);
        components.add(fpsLabel);
        components.add(configBox);
        components.add(groupBox);
        components.add(configButton);
        components.add(exposureButton);
        components.add(startButton);
        components.add(stopButton);
        components.add(roiButton);
        components.add(exposureField);
        components.add(roiWField);
        components.add(roiHField);
        components.add(roiXField);
        components.add(roiYField);
        components.add(queuingPanel);
        components.add(sendingPanel);
        disableControllers();

        buttons = new ArrayList<>();
        buttons.add(configButton);
        buttons.add(exposureButton);
        buttons.add(startButton);
        buttons.add(stopButton);
        buttons.add(roiButton);
        
        
    }
    
    void enablePanel(ControllerGui motherGui, String adress, int port, String channelName) {
        init();
        this.motherGui = motherGui;
        this.channelName = channelName;
        channelLabel.setText("Channel: " + channelName);
        if (adress != null) {
            client = new CameraClient(adress, port, this);
            client.start();
        }
    }
    
    void disablePanel() {
        init();
        disableControllers();
        this.setEnabled(false);
    }

    private void updateRoi() {
        sendInstruction("get roi");
        if (instructionDone) {
            int[] rois = client.roi;
            roiLabel.setText("ROI: " + rois[0] + ", " + rois[1] + ", " + rois[2] + ", " + rois[3]);
        }

    }

    private void updateExposure() {
        sendInstruction("get exposure");
        if (instructionDone) {
            exposureLabel.setText("Exposure Time: " + Math.round(client.exposure * 1000) / 1000.0);
        }
    }

    private void updateGroups() {
        sendInstruction("get groups");
        if (instructionDone) {
            String[] groups = client.getGroupArray();
            groupBox.removeAllItems();
            for (String s : groups) {
                groupBox.addItem(s);
            }
            updateConfigs(groupBox.getSelectedIndex());
        }
    }

    private void updateConfigs(int groupId) {
        String[] configs = client.getConfigArray(groupId);
        configBox.removeAllItems();
        for (String s : configs) {
            configBox.addItem(s);
        }
    }

    private void updateStatus() {
        sendInstruction("get status");
        if (instructionDone) {
            fpsLabel.setText("FPS: " + Math.round(client.fps * 10) / 10.0);
            if (client.queued) {
                queuingPanel.setBackground(Color.GREEN);
            } else {
                queuingPanel.setBackground(Color.RED);
            }
            if (client.sended) {
                sendingPanel.setBackground(Color.GREEN);
            } else {
                sendingPanel.setBackground(Color.RED);
            }
        }
    }

    void resetCamStatus() {
        if (updateThread != null) {
            updateThread.interrupt();
        }
        fpsLabel.setText("FPS: -");
        queuingPanel.setBackground(defaultColor);
        sendingPanel.setBackground(defaultColor);
    }

    void sendInstruction(String command) {
        instructionDone = true;
        client.addInstruction(command);
    }

    void enableButtons() {
        for (JButton b : buttons) {
            b.setEnabled(true);
        }
    }

    void disableButtons() {
        for (JButton b : buttons) {
            b.setEnabled(false);
        }
    }

    void enableControllers() {
        for (Component comp : components) {
            comp.setEnabled(true);
        }
    }

    void disableControllers() {
        for (Component comp : components) {
            comp.setEnabled(false);
        }
    }
    
    void startCam() {
        sendInstruction("start");
        if (instructionDone) {
            disableButtons();
            stopButton.setEnabled(true);
            updateThread = new StatusUpdateThread();
            updateThread.start();
        }
    }

    void stopCam() {
        sendInstruction("stop");
        if (instructionDone) {
            enableButtons();
            stopButton.setEnabled(false);
            resetCamStatus();
        }
    }

    void setRoi() {
        try {
            int[] roi = getRoi();
            String sRoi = Tool.encodeArray("set roi", roi);
            sendInstruction(sRoi);
            if (instructionDone) {
                updateRoi();
            }
        } catch (NumberFormatException ex) {
        }
    }

    void setExposureTime() {
        try {
            String exposureString = exposureField.getText();
            double exposureTime = Double.parseDouble(exposureString);
            sendInstruction("set exposure;" + exposureTime);
            if (instructionDone) {
                updateExposure();
            }
        } catch (NumberFormatException ex) {
        }
    }

    void setConfig() {
        int[] ids = new int[2];
        ids[0] = groupBox.getSelectedIndex();
        ids[1] = configBox.getSelectedIndex();
        String stringIds = Tool.encodeArray("set config", ids);
        sendInstruction(stringIds);
        if (instructionDone) {
            updateRoi();
            updateExposure();
        }
    }

    void groupBoxSelected() {
        int groupId = groupBox.getSelectedIndex();
        if (groupId >= 0) {
            updateConfigs(groupId);
        }
    }

    int[] getRoi() throws NumberFormatException {
        int[] roi = new int[ROILENGTH];
        roi[0] = Integer.parseInt(roiXField.getText());
        roi[1] = Integer.parseInt(roiYField.getText());
        roi[2] = Integer.parseInt(roiWField.getText());
        roi[3] = Integer.parseInt(roiHField.getText());
        return roi;
    }
    
    private class StatusUpdateThread extends Thread {
        @Override
        public void run() {
            while (!isInterrupted()) {
                try {
                    sleep(UPDATEDELAY);
                    updateStatus();
                } catch (InterruptedException ex) {
                    interrupt();
                }
            }
        }
    }

    @Override
    public void showText(String text) {
        motherGui.showText(text);
    }

    @Override
    public void registerClient() {
        updateRoi();
        updateExposure();
        updateGroups();
        enableControllers();
        stopButton.setEnabled(false);
    }

    @Override
    public void unregisterClient() {
        disableControllers();
        resetCamStatus();
    }
    
    @Override
    public void handleError(String error) {
        instructionDone = false;
        showText("Gui: Error with camera channel '" + channelName + "' occurred");
        showText(error);
    }

    @Override
    public void interruptInstruction() {
        instructionDone = false;
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        channelLabel = new javax.swing.JLabel();
        roiLabel = new javax.swing.JLabel();
        exposureLabel = new javax.swing.JLabel();
        startButton = new javax.swing.JButton();
        stopButton = new javax.swing.JButton();
        fpsLabel = new javax.swing.JLabel();
        queuingPanel = new javax.swing.JPanel();
        queuingLabel = new javax.swing.JLabel();
        sendingPanel = new javax.swing.JPanel();
        sendingLabel = new javax.swing.JLabel();
        roiXField = new javax.swing.JTextField();
        roiYField = new javax.swing.JTextField();
        roiWField = new javax.swing.JTextField();
        roiHField = new javax.swing.JTextField();
        roiButton = new javax.swing.JButton();
        exposureField = new javax.swing.JTextField();
        msLabel = new javax.swing.JLabel();
        exposureButton = new javax.swing.JButton();
        groupBox = new javax.swing.JComboBox<>();
        configBox = new javax.swing.JComboBox<>();
        configButton = new javax.swing.JButton();

        setBorder(javax.swing.BorderFactory.createTitledBorder("Camera"));

        channelLabel.setText("Channel: - ");

        roiLabel.setText("ROI: -");

        exposureLabel.setText("Exposure Time: - ");

        startButton.setText("Start Acquisition");
        startButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                startButtonActionPerformed(evt);
            }
        });

        stopButton.setText("Stop Acquisition");
        stopButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                stopButtonActionPerformed(evt);
            }
        });

        fpsLabel.setText("FPS: -");

        queuingPanel.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        queuingLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        queuingLabel.setText("Image Queuing");

        javax.swing.GroupLayout queuingPanelLayout = new javax.swing.GroupLayout(queuingPanel);
        queuingPanel.setLayout(queuingPanelLayout);
        queuingPanelLayout.setHorizontalGroup(
            queuingPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(queuingPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(queuingLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        queuingPanelLayout.setVerticalGroup(
            queuingPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(queuingLabel)
        );

        sendingPanel.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        sendingLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        sendingLabel.setText("Image Sending");

        javax.swing.GroupLayout sendingPanelLayout = new javax.swing.GroupLayout(sendingPanel);
        sendingPanel.setLayout(sendingPanelLayout);
        sendingPanelLayout.setHorizontalGroup(
            sendingPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(sendingPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(sendingLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        sendingPanelLayout.setVerticalGroup(
            sendingPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(sendingLabel)
        );

        roiXField.setText("765");
        roiXField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                roiXFieldKeyPressed(evt);
            }
        });

        roiYField.setText("765");
        roiYField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                roiYFieldKeyPressed(evt);
            }
        });

        roiWField.setText("520");
        roiWField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                roiWFieldKeyPressed(evt);
            }
        });

        roiHField.setText("520");
        roiHField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                roiHFieldKeyPressed(evt);
            }
        });

        roiButton.setText("Set ROI");
        roiButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                roiButtonActionPerformed(evt);
            }
        });

        exposureField.setText("3.509");
        exposureField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                exposureFieldKeyPressed(evt);
            }
        });

        msLabel.setText("ms");

        exposureButton.setText("Set Exposure");
        exposureButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                exposureButtonActionPerformed(evt);
            }
        });

        groupBox.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                groupBoxItemStateChanged(evt);
            }
        });

        configButton.setText("Set Config");
        configButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                configButtonActionPerformed(evt);
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
                        .addComponent(roiXField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(roiYField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(roiWField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(roiHField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(roiButton)
                        .addGap(18, 18, 18)
                        .addComponent(exposureField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(msLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(exposureButton)
                        .addGap(18, 18, 18)
                        .addComponent(groupBox, javax.swing.GroupLayout.PREFERRED_SIZE, 120, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(configBox, javax.swing.GroupLayout.PREFERRED_SIZE, 120, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(configButton)
                        .addGap(0, 27, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(fpsLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addGap(18, 18, 18)
                                .addComponent(queuingPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addGap(18, 18, 18)
                                .addComponent(sendingPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addGap(0, 0, Short.MAX_VALUE))
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(channelLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addGap(18, 18, 18)
                                .addComponent(roiLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addGap(18, 18, 18)
                                .addComponent(exposureLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addGap(86, 86, 86)
                                .addComponent(startButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(stopButton)))
                        .addContainerGap())))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(channelLabel)
                    .addComponent(roiLabel)
                    .addComponent(stopButton)
                    .addComponent(startButton)
                    .addComponent(exposureLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(fpsLabel)
                    .addComponent(queuingPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(sendingPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(roiButton)
                    .addComponent(roiXField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(roiYField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(roiWField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(roiHField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(exposureButton)
                    .addComponent(exposureField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(msLabel)
                    .addComponent(groupBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(configBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(configButton))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void startButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_startButtonActionPerformed
        startCam();
    }//GEN-LAST:event_startButtonActionPerformed

    private void stopButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_stopButtonActionPerformed
        stopCam();
    }//GEN-LAST:event_stopButtonActionPerformed

    private void roiButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_roiButtonActionPerformed
        setRoi();
    }//GEN-LAST:event_roiButtonActionPerformed

    private void exposureButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_exposureButtonActionPerformed
        setExposureTime();
    }//GEN-LAST:event_exposureButtonActionPerformed

    private void groupBoxItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_groupBoxItemStateChanged
        groupBoxSelected();
    }//GEN-LAST:event_groupBoxItemStateChanged

    private void configButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_configButtonActionPerformed
        setConfig();
    }//GEN-LAST:event_configButtonActionPerformed

    private void roiXFieldKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_roiXFieldKeyPressed
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            setRoi();
        }
    }//GEN-LAST:event_roiXFieldKeyPressed

    private void roiYFieldKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_roiYFieldKeyPressed
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            setRoi();
        }
    }//GEN-LAST:event_roiYFieldKeyPressed

    private void roiWFieldKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_roiWFieldKeyPressed
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            setRoi();
        }
    }//GEN-LAST:event_roiWFieldKeyPressed

    private void roiHFieldKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_roiHFieldKeyPressed
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            setRoi();
        }
    }//GEN-LAST:event_roiHFieldKeyPressed

    private void exposureFieldKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_exposureFieldKeyPressed
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            setExposureTime();
        }
    }//GEN-LAST:event_exposureFieldKeyPressed


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel channelLabel;
    private javax.swing.JComboBox<String> configBox;
    private javax.swing.JButton configButton;
    private javax.swing.JButton exposureButton;
    private javax.swing.JTextField exposureField;
    private javax.swing.JLabel exposureLabel;
    private javax.swing.JLabel fpsLabel;
    private javax.swing.JComboBox<String> groupBox;
    private javax.swing.JLabel msLabel;
    private javax.swing.JLabel queuingLabel;
    private javax.swing.JPanel queuingPanel;
    private javax.swing.JButton roiButton;
    private javax.swing.JTextField roiHField;
    private javax.swing.JLabel roiLabel;
    private javax.swing.JTextField roiWField;
    private javax.swing.JTextField roiXField;
    private javax.swing.JTextField roiYField;
    private javax.swing.JLabel sendingLabel;
    private javax.swing.JPanel sendingPanel;
    private javax.swing.JButton startButton;
    private javax.swing.JButton stopButton;
    // End of variables declaration//GEN-END:variables
}
