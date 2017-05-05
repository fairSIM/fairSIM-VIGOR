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
import java.util.ArrayList;
import java.util.List;
import javax.swing.DefaultListModel;
import org.fairsim.livemode.LiveControlPanel;

/**
 * Easy to use controller gui to controll the fast sim setup
 * @author Mario
 */
public class EasyGui extends javax.swing.JPanel {

    private final LiveControlPanel lcp;
    private final AdvGui advanced;
    private final Ctrl controller;
    private final Sync sync;
    private final Reg regPanel;
    private final  List<Cam> camGuis;
    private final List<RunningOrder> runningOrders = new ArrayList<>();
    private final List<RunningOrder> possibleRos = new ArrayList<>();
    private int delay = 0;
    
    /**
     * Creates new form EasyGui
     */
    public EasyGui(LiveControlPanel lcp, AdvGui advGui) {
        initComponents();
        this.lcp = lcp;
        advanced = advGui;
        controller = advGui.getCtrl();
        sync = advGui.getSync();
        regPanel = advGui.getReg();
        camGuis = advGui.getCams();
    }
    
    /**
     * activates this if all necessary devices are connected, connects arduino
     * & device running orders
     */
    void activate() {
        if(!connected()) return;
        ControllerClient.ArduinoRunningOrder[] arduinoRos = controller.getArduinoRos();
        String[] deviceRos = controller.getDeviceRos();
        runningOrders.clear();
        for(int i = 0; i < arduinoRos.length; i++) {
            String device = arduinoRos[i].name.split("_", 2)[0];
            String name = arduinoRos[i].name.split("_", 2)[1];
            int deviceRo = getDeviceRo(name, deviceRos);
            if (deviceRo < 0) continue;
            boolean allowBigRoi = name.split("_")[1].endsWith("ms");
            runningOrders.add(new RunningOrder(device, name, deviceRo, i,
                    arduinoRos[i].syncDelay, arduinoRos[i].syncFreq,
                    arduinoRos[i].exposureTime, allowBigRoi));
        }
        enableLaserPanel();
    }
    
    /**
     * finds the id of the device running order with a specific name
     * @param name name of the running order
     * @param deviceRos 
     * @return id of the running order, -1 if no running order was found
     */
    private int getDeviceRo(String name, String[] deviceRos) {
        for(int i = 0; i < deviceRos.length; i++) {
            if (name.equals(deviceRos[i])) return i;
        }
        return -1;
    }

    /**
     * class to handle running orders for the whole fast sim setup
     */
    class RunningOrder {
        final String device, name;
        final int deviceRo, arduinoRo;
        final int syncDelay, syncFreq;
        final int exposureTime;
        final int colorCount;
        final String camGroup = "fastSIM", camConfig = "fastSIM";
        final String illuminationTime;
        final boolean allowBigRoi;
        final char[] colors;

        RunningOrder(String device, String name, int deviceRo, int arduinoRo, int sDelay, int sFreq, int eTime, boolean allowBigRoi) {
            this.device = device;
            this.name = name;
            this.deviceRo = deviceRo;
            this.arduinoRo = arduinoRo;
            this.syncDelay = sDelay;
            this.syncFreq = sFreq;
            this.exposureTime = eTime;
            this.allowBigRoi = allowBigRoi;
            colorCount = Integer.parseInt(name.split("col")[0]);
            colors = new char[colorCount];
            illuminationTime = name.split("_")[1];
        }
    }
    
    /**
     * interface for the advanced gui
     */
    static interface AdvGui {
        
        /**
         * sets all options for this running order
         * @param ro running order
         * @throws org.fairsim.controller.EasyGui.EasyGuiException if anything
         * went wrong
         */
        void setRo(RunningOrder ro) throws EasyGuiException;
        
        /**
         * 
         * @return controller gui of the advanced gui
         */
        Ctrl getCtrl();
        
        /**
         * 
         * @return sync gui of the advanced gui
         */
        Sync getSync();
        
        /**
         * 
         * @return registration gui of the advanced gui
         */
        Reg getReg();
        
        /**
         * 
         * @return list of camnera guis of the advanced gui
         */
        List<Cam> getCams();
    }
    
    /**
     * interface for the controller gui of arduino & device
     */
    static interface Ctrl {
        
        /**
         * preparations for the easy gui
         * @throws org.fairsim.controller.EasyGui.EasyGuiException anything went
         * wrong
         */
        void enableEasy() throws EasyGui.EasyGuiException;
        
        /**
         * sets all options for this running order
         * @param ro running order
         * @throws org.fairsim.controller.EasyGui.EasyGuiException if anything
         * went wrong
         */
        void setRo(RunningOrder ro) throws EasyGuiException;
        
        /**
         * 
         * @return running orders of the arduino
         */
        ControllerClient.ArduinoRunningOrder[] getArduinoRos();
        
        /**
         * 
         * @return running orders of the device
         */
        String[] getDeviceRos();
        
        /**
         * sets the delay between single sim images in movie mode
         * @param delay delay in milliseconds
         */
        void setDelay(int delay);
        
        /**
         * starts the sim movie mode
         */
        void startMovie();
        
        /**
         * stops the sim movie mode
         */
        void stopMovie();
        
        /**
         * takes a sim photo
         */
        void takePhoto();
    }
    
    /**
     * interface for the sync gui
     */
    static interface Sync {
        /**
         * sets all options for this running order
         * @param ro running order
         * @throws org.fairsim.controller.EasyGui.EasyGuiException if anything
         * went wrong
         */
        void setRo(RunningOrder ro) throws EasyGuiException;
    }
    
    /**
     * interface for the registration gui
     */
    static interface Reg {
        /**
         * (de)activates registration
         * @param b activate/deactivate for true/false
         */
        void register(boolean b);
    }
    
    /**
     * interface for the camera guis
     */
    static interface Cam {
        
        /**
         * preparations for the easy gui
         * @throws org.fairsim.controller.EasyGui.EasyGuiException anything went
         * wrong
         */
        void enableEasy() throws EasyGuiException;
        
        /**
         * sets all options for this running order
         * @param ro running order
         * @throws org.fairsim.controller.EasyGui.EasyGuiException if anything
         * went wrong
         */
        void setRo(RunningOrder ro) throws EasyGuiException;
        
        /**
         * starts camera acquisition
         */
        void startMovie();
        
        /**
         * stops camera acquisition
         */
        void stopMovie();
    }
    
    /**
     * exception class for the easy to use gui
     */
    static class EasyGuiException extends Exception {
        EasyGuiException(String message) {
            super(message);
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

        laserPanel = new javax.swing.JPanel();
        blueCheckBox = new javax.swing.JCheckBox();
        greenCheckBox = new javax.swing.JCheckBox();
        redCheckBox = new javax.swing.JCheckBox();
        itPanel = new javax.swing.JPanel();
        jScrollPane1 = new javax.swing.JScrollPane();
        itList = new javax.swing.JList<>();
        controlPanel = new javax.swing.JPanel();
        runButton = new javax.swing.JToggleButton();
        photoButton = new javax.swing.JButton();
        registrationButton = new javax.swing.JToggleButton();
        delaySlider = new javax.swing.JSlider();
        delayLabel = new javax.swing.JLabel();
        msLabel = new javax.swing.JLabel();
        paramButton = new javax.swing.JButton();
        enableButton = new javax.swing.JButton();
        statusPanel = new javax.swing.JPanel();
        statusLabel = new javax.swing.JLabel();

        laserPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Laser Colors"));
        laserPanel.setEnabled(false);

        blueCheckBox.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        blueCheckBox.setForeground(new java.awt.Color(0, 0, 255));
        blueCheckBox.setText("488 nm");
        blueCheckBox.setEnabled(false);
        blueCheckBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                blueCheckBoxActionPerformed(evt);
            }
        });

        greenCheckBox.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        greenCheckBox.setForeground(new java.awt.Color(0, 150, 0));
        greenCheckBox.setText("568 nm");
        greenCheckBox.setEnabled(false);
        greenCheckBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                greenCheckBoxActionPerformed(evt);
            }
        });

        redCheckBox.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        redCheckBox.setForeground(new java.awt.Color(255, 0, 0));
        redCheckBox.setText("647 nm");
        redCheckBox.setEnabled(false);
        redCheckBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                redCheckBoxActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout laserPanelLayout = new javax.swing.GroupLayout(laserPanel);
        laserPanel.setLayout(laserPanelLayout);
        laserPanelLayout.setHorizontalGroup(
            laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(laserPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(blueCheckBox)
                    .addComponent(greenCheckBox)
                    .addComponent(redCheckBox))
                .addContainerGap(31, Short.MAX_VALUE))
        );
        laserPanelLayout.setVerticalGroup(
            laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(laserPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(blueCheckBox)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(greenCheckBox)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(redCheckBox)
                .addContainerGap(147, Short.MAX_VALUE))
        );

        itPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Illumination Time"));
        itPanel.setEnabled(false);

        itList.setModel(new DefaultListModel<String>());
        itList.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_SELECTION);
        itList.setEnabled(false);
        itList.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                itListMouseClicked(evt);
            }
        });
        jScrollPane1.setViewportView(itList);

        javax.swing.GroupLayout itPanelLayout = new javax.swing.GroupLayout(itPanel);
        itPanel.setLayout(itPanelLayout);
        itPanelLayout.setHorizontalGroup(
            itPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(itPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 118, Short.MAX_VALUE)
                .addContainerGap())
        );
        itPanelLayout.setVerticalGroup(
            itPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(itPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jScrollPane1)
                .addContainerGap())
        );

        controlPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Control"));
        controlPanel.setEnabled(false);

        runButton.setText("Run");
        runButton.setEnabled(false);
        runButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                runButtonActionPerformed(evt);
            }
        });

        photoButton.setText("Take a photo");
        photoButton.setEnabled(false);
        photoButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                photoButtonActionPerformed(evt);
            }
        });

        registrationButton.setText("Image Registration");
        registrationButton.setEnabled(false);
        registrationButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                registrationButtonActionPerformed(evt);
            }
        });

        delaySlider.setMaximum(20);
        delaySlider.setValue(0);
        delaySlider.setEnabled(false);
        delaySlider.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                delaySliderStateChanged(evt);
            }
        });

        delayLabel.setText("Delay");
        delayLabel.setEnabled(false);

        msLabel.setHorizontalAlignment(javax.swing.SwingConstants.RIGHT);
        msLabel.setText("0 ms");
        msLabel.setEnabled(false);

        paramButton.setText("Parameter Estimation");
        paramButton.setEnabled(false);
        paramButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                paramButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout controlPanelLayout = new javax.swing.GroupLayout(controlPanel);
        controlPanel.setLayout(controlPanelLayout);
        controlPanelLayout.setHorizontalGroup(
            controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(controlPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(registrationButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(runButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(photoButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(controlPanelLayout.createSequentialGroup()
                        .addComponent(delayLabel)
                        .addGap(18, 18, 18)
                        .addComponent(delaySlider, javax.swing.GroupLayout.PREFERRED_SIZE, 0, Short.MAX_VALUE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(msLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 50, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addComponent(paramButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        controlPanelLayout.setVerticalGroup(
            controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(controlPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(registrationButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(delaySlider, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(delayLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(msLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 26, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(runButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(photoButton)
                .addGap(18, 18, 18)
                .addComponent(paramButton)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        enableButton.setText("Enable");
        enableButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                enableButtonActionPerformed(evt);
            }
        });

        statusPanel.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        statusLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        statusLabel.setText("Status Line");

        javax.swing.GroupLayout statusPanelLayout = new javax.swing.GroupLayout(statusPanel);
        statusPanel.setLayout(statusPanelLayout);
        statusPanelLayout.setHorizontalGroup(
            statusPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(statusPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(statusLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        statusPanelLayout.setVerticalGroup(
            statusPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(statusLabel, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(laserPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(itPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(controlPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(enableButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(statusPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(statusPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(enableButton, javax.swing.GroupLayout.DEFAULT_SIZE, 27, Short.MAX_VALUE))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(controlPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(itPanel, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(laserPanel, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void blueCheckBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_blueCheckBoxActionPerformed
        disableControlPanel();
        enableItPanel();
    }//GEN-LAST:event_blueCheckBoxActionPerformed

    private void greenCheckBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_greenCheckBoxActionPerformed
        disableControlPanel();
        enableItPanel();
    }//GEN-LAST:event_greenCheckBoxActionPerformed

    private void redCheckBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_redCheckBoxActionPerformed
        disableControlPanel();
        enableItPanel();
    }//GEN-LAST:event_redCheckBoxActionPerformed

    private void delaySliderStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_delaySliderStateChanged
        delay = delaySlider.getValue() * 50;
        msLabel.setText(delay + " ms");
        controller.setDelay(delay);
    }//GEN-LAST:event_delaySliderStateChanged

    private void enableButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_enableButtonActionPerformed
        activate();
    }//GEN-LAST:event_enableButtonActionPerformed

    private void itListMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_itListMouseClicked
        try {
            setRo();
            enableControlPanel();
        } catch (EasyGuiException ex) {
            setStatus(ex.getMessage(), true);
        }
    }//GEN-LAST:event_itListMouseClicked

    private void runButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_runButtonActionPerformed
        if (runButton.isSelected()) {
            lcp.getSequenceExtractor().clearBuffers();
            disableControllers();
            for(Cam c : camGuis) {
                c.startMovie();
            }
            controller.startMovie();
            setStatus("Capturing images started");
        } else {
            for(Cam c : camGuis) {
                c.stopMovie();
            }
            controller.stopMovie();
            setStatus("Capturing images stopped");
            enableControllers();
        }
        registrationButton.setEnabled(true);
        runButton.setEnabled(true);
        paramButton.setEnabled(true);
    }//GEN-LAST:event_runButtonActionPerformed

    private void photoButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_photoButtonActionPerformed
        lcp.getSequenceExtractor().clearBuffers();
        for(Cam c : camGuis) {
            c.startMovie();
        }
        controller.takePhoto();
        for(Cam c : camGuis) {
            c.stopMovie();
        }
        setStatus("Captured a photo");
    }//GEN-LAST:event_photoButtonActionPerformed

    private void registrationButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_registrationButtonActionPerformed
        boolean registering = registrationButton.isSelected();
        regPanel.register(registering);
        if (registering) setStatus("Activated image registration");
        else setStatus("Deactivated image registration");
    }//GEN-LAST:event_registrationButtonActionPerformed

    private void paramButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_paramButtonActionPerformed
        lcp.doParamEstimation();
        setStatus("Parameter estimation");
    }//GEN-LAST:event_paramButtonActionPerformed

    /**
     * asks if all necessary devices are connected, if not disables the easy gui
     * @return true if all necessary devices are connected, else false
     */
    private boolean connected() {
        try {
            controller.enableEasy();
            for (Cam c : camGuis) {
                c.enableEasy();
            }
            return true;
        } catch (EasyGuiException ex) {
            disableLaserPanel();
            disableItPanel();
            disableControlPanel();
            setStatus(ex.getMessage(), true);
            return false;
        }
    }
    
    /**
     * disables all controllers, called after starting the movie mode
     */
    private void disableControllers() {
        disableLaserPanel();
        disableItPanel();
        disableControlPanel();
        enableButton.setEnabled(false);
    }
    
    /**
     * enables all controllers, called after stopping the movie mode
     */
    private void enableControllers() {
        enableLaserPanel();
        enableItPanel();
        enableControlPanel();
        enableButton.setEnabled(true);
    }
    
    /**
     * enables the laser panel
     */
    private void enableLaserPanel() {
        if (!connected()) return;
        laserPanel.setEnabled(true);
        redCheckBox.setEnabled(true);
        greenCheckBox.setEnabled(true);
        blueCheckBox.setEnabled(true);
        setStatus("Choose laser colors");
    }
    
    /**
     * disable the laser panel
     */
    private void disableLaserPanel() {
        laserPanel.setEnabled(false);
        redCheckBox.setEnabled(false);
        greenCheckBox.setEnabled(false);
        blueCheckBox.setEnabled(false);
    }
    
    /**
     * enables the illumination time panel
     */
    private void enableItPanel() {
        if (!connected()) return;
        updateItList();
        itPanel.setEnabled(true);
        itList.setEnabled(true);
        setStatus("Choose illumination time");
    }
    
    /**
     * updates the illumination time list
     */
    private void updateItList() {
        List<String> colors = new ArrayList<>();
        if (redCheckBox.isSelected()) colors.add("r");
        if (greenCheckBox.isSelected()) colors.add("g");
        if (blueCheckBox.isSelected()) colors.add("b");
        possibleRos.clear();
        int colorCount = colors.size();
        // algorithm to find the correct running orders for the illumination time list
        for (int i = 0; i < runningOrders.size(); i++) {
            if (Integer.parseInt(runningOrders.get(i).name.split("col")[0]) == colorCount) {
                switch (colorCount) {
                    case 3: {
                        possibleRos.add(runningOrders.get(i));
                        break;
                    }
                    case 2: {
                        if (runningOrders.get(i).name.endsWith("_" + colors.get(0) + colors.get(1))) possibleRos.add(runningOrders.get(i));
                        break;
                    }
                    case 1: {
                        if (runningOrders.get(i).name.endsWith("_" + colors.get(0))) possibleRos.add(runningOrders.get(i));
                        break;
                    }
                    default: {
                        throw new RuntimeException("Only 1, 2 or 3 colors allowed");
                    }
                }
            }
        }
        DefaultListModel<String> model = (DefaultListModel<String>) itList.getModel();
        model.clear();
        for (RunningOrder ro : possibleRos) {
            model.addElement(ro.illuminationTime);
        }
    }
    
    /**
     * disables the illumination time panel
     */
    private void disableItPanel() {
        itPanel.setEnabled(false);
        itList.setEnabled(false);
    }
    
    /**
     * enables the control panel
     */
    private void enableControlPanel() {
        if (!connected()) return;
        controlPanel.setEnabled(true);
        registrationButton.setEnabled(true);
        delayLabel.setEnabled(true);
        delaySlider.setEnabled(true);
        msLabel.setEnabled(true);
        runButton.setEnabled(true);
        photoButton.setEnabled(true);
        paramButton.setEnabled(true);
    }
    
    /**
     * disables the control panel
     */
    private void disableControlPanel() {
        controlPanel.setEnabled(false);
        registrationButton.setEnabled(false);
        delayLabel.setEnabled(false);
        delaySlider.setEnabled(false);
        msLabel.setEnabled(false);
        runButton.setEnabled(false);
        photoButton.setEnabled(false);
        paramButton.setEnabled(false);
    }
    
    /**
     * sets the status line of this with green text
     * @param message message to be shown
     */
    private void setStatus(String message) {
        setStatus(message, false);
    }
    
    /**
     * sets the status line of this with red/green text
     * @param message message to be shown
     * @param error red/green for true/false
     */
    private void setStatus(String message, boolean error) {
        statusLabel.setText(message);
        if (error) statusLabel.setForeground(new Color(255, 0, 0));
        else statusLabel.setForeground(new Color(0, 150, 0));
    }
    
    /**
     * sets the options for the selected running order at all devices
     * @throws org.fairsim.controller.EasyGui.EasyGuiException if anything went
     * wrong
     */
    private void setRo() throws EasyGuiException {
        RunningOrder ro = possibleRos.get(itList.getSelectedIndex());
        advanced.setRo(ro);
        controller.setRo(ro);
        sync.setRo(ro);
        for (Cam c : camGuis) {
            c.setRo(ro);
        }
        setStatus("Running order: " + ro.device + "_" + ro.name);
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JCheckBox blueCheckBox;
    private javax.swing.JPanel controlPanel;
    private javax.swing.JLabel delayLabel;
    private javax.swing.JSlider delaySlider;
    private javax.swing.JButton enableButton;
    private javax.swing.JCheckBox greenCheckBox;
    private javax.swing.JList<String> itList;
    private javax.swing.JPanel itPanel;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JPanel laserPanel;
    private javax.swing.JLabel msLabel;
    private javax.swing.JButton paramButton;
    private javax.swing.JButton photoButton;
    private javax.swing.JCheckBox redCheckBox;
    private javax.swing.JToggleButton registrationButton;
    private javax.swing.JToggleButton runButton;
    private javax.swing.JLabel statusLabel;
    private javax.swing.JPanel statusPanel;
    // End of variables declaration//GEN-END:variables
}
