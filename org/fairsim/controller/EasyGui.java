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
 *
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
    
    void activate() {
        if(!connected()) return;
        ControllerClient.ArduinoRunningOrder[] arduinoRos = null;
        String[] deviceRos = null;
        arduinoRos = controller.getArduinoRos();
        deviceRos = controller.getDeviceRos();
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
    
    private int getDeviceRo(String name, String[] deviceRos) {
        for(int i = 0; i < deviceRos.length; i++) {
            if (name.equals(deviceRos[i])) return i;
        }
        return -1;
    }

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
    
    static interface AdvGui {
        void setRo(RunningOrder ro) throws EasyGuiException;
        Ctrl getCtrl();
        Sync getSync();
        Reg getReg();
        List<Cam> getCams();
    }
    
    static interface Ctrl {
        void enableEasy() throws EasyGui.EasyGuiException;
        void setRo(RunningOrder ro) throws EasyGuiException;
        ControllerClient.ArduinoRunningOrder[] getArduinoRos();
        String[] getDeviceRos();
        void setDelay(int delay);
        void startMovie();
        void stopMovie();
        void takePhoto();
    }
    
    static interface Sync {
        void setRo(RunningOrder ro) throws EasyGuiException;
    }
    
    static interface Reg {
        void register(boolean b);
    }
    
    static interface Cam {
        void enableEasy() throws EasyGuiException;
        void setRo(RunningOrder ro) throws EasyGuiException;
        void startMovie();
        void stopMovie();
    }
    
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
        controllPanel = new javax.swing.JPanel();
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

        controllPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Controll"));
        controllPanel.setEnabled(false);

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

        javax.swing.GroupLayout controllPanelLayout = new javax.swing.GroupLayout(controllPanel);
        controllPanel.setLayout(controllPanelLayout);
        controllPanelLayout.setHorizontalGroup(
            controllPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(controllPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(controllPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(registrationButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(runButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(photoButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(controllPanelLayout.createSequentialGroup()
                        .addComponent(delayLabel)
                        .addGap(18, 18, 18)
                        .addComponent(delaySlider, javax.swing.GroupLayout.PREFERRED_SIZE, 0, Short.MAX_VALUE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(msLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 50, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addComponent(paramButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        controllPanelLayout.setVerticalGroup(
            controllPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(controllPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(registrationButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(controllPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
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
                        .addComponent(controllPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
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
                    .addComponent(controllPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(itPanel, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(laserPanel, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void blueCheckBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_blueCheckBoxActionPerformed
        disableControllPanel();
        enableItPanel();
    }//GEN-LAST:event_blueCheckBoxActionPerformed

    private void greenCheckBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_greenCheckBoxActionPerformed
        disableControllPanel();
        enableItPanel();
    }//GEN-LAST:event_greenCheckBoxActionPerformed

    private void redCheckBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_redCheckBoxActionPerformed
        disableControllPanel();
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
            enableControllPanel();
        } catch (EasyGuiException ex) {
            setStatus(ex.getMessage(), true);
        }
    }//GEN-LAST:event_itListMouseClicked

    private void runButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_runButtonActionPerformed
        if (runButton.isSelected()) {
            disableGui();
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
            enableGui();
        }
        registrationButton.setEnabled(true);
        runButton.setEnabled(true);
        paramButton.setEnabled(true);
    }//GEN-LAST:event_runButtonActionPerformed

    private void photoButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_photoButtonActionPerformed
        for(Cam c : camGuis) {
            c.startMovie();
        }
        controller.takePhoto();
        for(Cam c : camGuis) {
            c.stopMovie();
        }
        setStatus("Cptured a photo");
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
            disableControllPanel();
            setStatus(ex.getMessage(), true);
            return false;
        }
    }
    
    private void disableGui() {
        disableLaserPanel();
        disableItPanel();
        disableControllPanel();
        enableButton.setEnabled(false);
    }
    
    private void enableGui() {
        enableLaserPanel();
        enableItPanel();
        enableControllPanel();
        enableButton.setEnabled(true);
    }
    
    private void enableLaserPanel() {
        if (!connected()) return;
        laserPanel.setEnabled(true);
        redCheckBox.setEnabled(true);
        greenCheckBox.setEnabled(true);
        blueCheckBox.setEnabled(true);
        setStatus("Choose laser colors");
    }
    
    private void disableLaserPanel() {
        laserPanel.setEnabled(false);
        redCheckBox.setEnabled(false);
        greenCheckBox.setEnabled(false);
        blueCheckBox.setEnabled(false);
    }
    
    private void enableItPanel() {
        if (!connected()) return;
        updateItList();
        itPanel.setEnabled(true);
        itList.setEnabled(true);
        setStatus("Choose illumination time");
    }
    
    private void updateItList() {
        List<String> colors = new ArrayList<>();
        if (redCheckBox.isSelected()) colors.add("r");
        if (greenCheckBox.isSelected()) colors.add("g");
        if (blueCheckBox.isSelected()) colors.add("b");
        possibleRos.clear();
        int colorCount = colors.size();
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
    
    private void disableItPanel() {
        itPanel.setEnabled(false);
        itList.setEnabled(false);
    }
    
    private void enableControllPanel() {
        if (!connected()) return;
        controllPanel.setEnabled(true);
        registrationButton.setEnabled(true);
        delayLabel.setEnabled(true);
        delaySlider.setEnabled(true);
        msLabel.setEnabled(true);
        runButton.setEnabled(true);
        photoButton.setEnabled(true);
        paramButton.setEnabled(true);
    }
    
    private void disableControllPanel() {
        controllPanel.setEnabled(false);
        registrationButton.setEnabled(false);
        delayLabel.setEnabled(false);
        delaySlider.setEnabled(false);
        msLabel.setEnabled(false);
        runButton.setEnabled(false);
        photoButton.setEnabled(false);
        paramButton.setEnabled(false);
    }
    
    private void setStatus(String message) {
        setStatus(message, false);
    }
    
    private void setStatus(String message, boolean error) {
        statusLabel.setText(message);
        if (error) statusLabel.setForeground(new Color(255, 0, 0));
        else statusLabel.setForeground(new Color(0, 150, 0));
    }
    
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
    private javax.swing.JPanel controllPanel;
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
