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
import java.awt.event.KeyEvent;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.DefaultListModel;
import org.fairsim.livemode.LiveControlPanel;
import org.fairsim.utils.Tool;

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
    private int runDelay = 0;
    private int runRecDelay = 0;
    private final String dyeFile;
    private int illuminationTime = -1;
    private int delayTime = -1;
    private int syncDelayTime = -1;
    private int syncFreq = -1;
    
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
        new File(System.getProperty("user.home") + "/documents/").mkdirs();
        dyeFile = System.getProperty("user.home") + "/documents/fairsim-dyes.txt";
        try {
            new File(dyeFile).createNewFile();
        } catch (IOException ex) {
            throw new RuntimeException("this should never happen");
        }
        updateDyeBoxes();
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
            if(device.equals("dmd") && controller.getType().equals("DMD") ||
                    device.equals("slm") && controller.getType().equals("FLCOS")) {
                String name = arduinoRos[i].name.split("_", 2)[1];
                int deviceRo = getDeviceRo(name, deviceRos);
                if (deviceRo < 0) continue;
                boolean allowBigRoi = name.split("_")[1].endsWith("ms");
                runningOrders.add(new RunningOrder(device, name, deviceRo, i,
                        arduinoRos[i].syncDelay, arduinoRos[i].syncFreq,
                        arduinoRos[i].exposureTime, allowBigRoi));
            }
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
        final String camGroup = "SIM", camConfig = "SIM";
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
        
        String getType();
        
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
        
        /**
         * sets the sync frequence
         * @param freq 
         */
        void setFreq(int freq);
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
        blueComboBox = new javax.swing.JComboBox<>();
        blueTextField = new javax.swing.JTextField();
        blueLabel = new javax.swing.JLabel();
        greenTextField = new javax.swing.JTextField();
        greenLabel = new javax.swing.JLabel();
        greenComboBox = new javax.swing.JComboBox<>();
        redTextField = new javax.swing.JTextField();
        redLabel = new javax.swing.JLabel();
        redComboBox = new javax.swing.JComboBox<>();
        blueDyeLabel = new javax.swing.JLabel();
        greenDyeLabel = new javax.swing.JLabel();
        redDyeLabel = new javax.swing.JLabel();
        dyeTextField = new javax.swing.JTextField();
        addDyeButton = new javax.swing.JButton();
        itPanel = new javax.swing.JPanel();
        jScrollPane1 = new javax.swing.JScrollPane();
        itList = new javax.swing.JList<>();
        controlPanel = new javax.swing.JPanel();
        runButton = new javax.swing.JToggleButton();
        photoButton = new javax.swing.JButton();
        registrationButton = new javax.swing.JToggleButton();
        runDelaySlider = new javax.swing.JSlider();
        runDelayLabel = new javax.swing.JLabel();
        runMsLabel = new javax.swing.JLabel();
        paramButton = new javax.swing.JButton();
        runRecDelayLabel = new javax.swing.JLabel();
        runRecDelaySlider = new javax.swing.JSlider();
        runRecMsLabel = new javax.swing.JLabel();
        runRecButton = new javax.swing.JToggleButton();
        enableButton = new javax.swing.JButton();
        statusPanel = new javax.swing.JPanel();
        statusLabel = new javax.swing.JLabel();
        samplePanel = new javax.swing.JPanel();
        sampleTextField = new javax.swing.JTextField();
        jLabel1 = new javax.swing.JLabel();

        laserPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Laser Colors"));
        laserPanel.setEnabled(false);

        blueCheckBox.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        blueCheckBox.setForeground(new java.awt.Color(0, 0, 255));
        blueCheckBox.setText("blue");
        blueCheckBox.setEnabled(false);
        blueCheckBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                blueCheckBoxActionPerformed(evt);
            }
        });

        greenCheckBox.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        greenCheckBox.setForeground(new java.awt.Color(0, 150, 0));
        greenCheckBox.setText("green");
        greenCheckBox.setEnabled(false);
        greenCheckBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                greenCheckBoxActionPerformed(evt);
            }
        });

        redCheckBox.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        redCheckBox.setForeground(new java.awt.Color(255, 0, 0));
        redCheckBox.setText("red");
        redCheckBox.setEnabled(false);
        redCheckBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                redCheckBoxActionPerformed(evt);
            }
        });

        blueTextField.setText("0000");

        blueLabel.setText("mW");

        greenTextField.setText("0000");

        greenLabel.setText("mW");

        greenComboBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                greenComboBoxActionPerformed(evt);
            }
        });

        redTextField.setText("0000");

        redLabel.setText("mW");

        redComboBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                redComboBoxActionPerformed(evt);
            }
        });

        blueDyeLabel.setText("Dye: ");

        greenDyeLabel.setText("Dye: ");

        redDyeLabel.setText("Dye: ");

        dyeTextField.addKeyListener(new java.awt.event.KeyAdapter() {
            public void keyPressed(java.awt.event.KeyEvent evt) {
                dyeTextFieldKeyPressed(evt);
            }
        });

        addDyeButton.setText("Add dye");
        addDyeButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                addDyeButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout laserPanelLayout = new javax.swing.GroupLayout(laserPanel);
        laserPanel.setLayout(laserPanelLayout);
        laserPanelLayout.setHorizontalGroup(
            laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(laserPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(dyeTextField, javax.swing.GroupLayout.Alignment.TRAILING)
                    .addGroup(laserPanelLayout.createSequentialGroup()
                        .addComponent(blueDyeLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(blueComboBox, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addGroup(laserPanelLayout.createSequentialGroup()
                        .addComponent(greenDyeLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(greenComboBox, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addGroup(laserPanelLayout.createSequentialGroup()
                        .addComponent(redDyeLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(redComboBox, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addComponent(addDyeButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(laserPanelLayout.createSequentialGroup()
                        .addGroup(laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(laserPanelLayout.createSequentialGroup()
                                .addComponent(blueCheckBox, javax.swing.GroupLayout.PREFERRED_SIZE, 60, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(blueTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(blueLabel))
                            .addGroup(laserPanelLayout.createSequentialGroup()
                                .addComponent(redCheckBox, javax.swing.GroupLayout.PREFERRED_SIZE, 60, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(redTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(redLabel))
                            .addGroup(laserPanelLayout.createSequentialGroup()
                                .addComponent(greenCheckBox, javax.swing.GroupLayout.PREFERRED_SIZE, 60, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(greenTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(greenLabel)))
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );
        laserPanelLayout.setVerticalGroup(
            laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(laserPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(blueCheckBox)
                    .addComponent(blueTextField, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(blueLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(blueComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(blueDyeLabel))
                .addGap(18, 18, 18)
                .addGroup(laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(greenCheckBox)
                    .addComponent(greenTextField, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(greenLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(greenComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(greenDyeLabel))
                .addGap(18, 18, 18)
                .addGroup(laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(redCheckBox)
                    .addComponent(redTextField, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(redLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(laserPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(redComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(redDyeLabel))
                .addGap(18, 18, 18)
                .addComponent(dyeTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(addDyeButton)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
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

        runDelaySlider.setMaximum(20);
        runDelaySlider.setValue(0);
        runDelaySlider.setEnabled(false);
        runDelaySlider.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                runDelaySliderStateChanged(evt);
            }
        });

        runDelayLabel.setText("Delay");
        runDelayLabel.setEnabled(false);

        runMsLabel.setHorizontalAlignment(javax.swing.SwingConstants.RIGHT);
        runMsLabel.setText("0 ms");
        runMsLabel.setEnabled(false);

        paramButton.setText("Parameter Estimation");
        paramButton.setEnabled(false);
        paramButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                paramButtonActionPerformed(evt);
            }
        });

        runRecDelayLabel.setText("Delay");
        runRecDelayLabel.setEnabled(false);

        runRecDelaySlider.setMaximum(20);
        runRecDelaySlider.setValue(0);
        runRecDelaySlider.setEnabled(false);
        runRecDelaySlider.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                runRecDelaySliderStateChanged(evt);
            }
        });

        runRecMsLabel.setHorizontalAlignment(javax.swing.SwingConstants.RIGHT);
        runRecMsLabel.setText("0 ms");
        runRecMsLabel.setEnabled(false);

        runRecButton.setForeground(new java.awt.Color(255, 0, 0));
        runRecButton.setText("Run & Record");
        runRecButton.setEnabled(false);
        runRecButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                runRecButtonActionPerformed(evt);
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
                    .addComponent(photoButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(paramButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(controlPanelLayout.createSequentialGroup()
                        .addGroup(controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addComponent(runButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addGroup(javax.swing.GroupLayout.Alignment.LEADING, controlPanelLayout.createSequentialGroup()
                                .addComponent(runDelayLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(runDelaySlider, javax.swing.GroupLayout.DEFAULT_SIZE, 36, Short.MAX_VALUE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(runMsLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 50, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addGap(18, 18, 18)
                        .addGroup(controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(controlPanelLayout.createSequentialGroup()
                                .addComponent(runRecDelayLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(runRecDelaySlider, javax.swing.GroupLayout.DEFAULT_SIZE, 36, Short.MAX_VALUE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(runRecMsLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 50, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(runRecButton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
                .addContainerGap())
        );
        controlPanelLayout.setVerticalGroup(
            controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(controlPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(registrationButton)
                .addGap(18, 18, 18)
                .addGroup(controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                        .addComponent(runDelaySlider, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(runDelayLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addGroup(controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(runMsLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 26, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(runRecDelayLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                    .addGroup(controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                        .addComponent(runRecDelaySlider, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(runRecMsLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 26, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(controlPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(runButton)
                    .addComponent(runRecButton))
                .addGap(18, 18, 18)
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

        samplePanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Sample"));

        jLabel1.setText("Sample description: ");

        javax.swing.GroupLayout samplePanelLayout = new javax.swing.GroupLayout(samplePanel);
        samplePanel.setLayout(samplePanelLayout);
        samplePanelLayout.setHorizontalGroup(
            samplePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, samplePanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jLabel1)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(sampleTextField)
                .addContainerGap())
        );
        samplePanelLayout.setVerticalGroup(
            samplePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(samplePanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(samplePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(sampleTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel1))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(samplePanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(laserPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
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
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(laserPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(itPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(controlPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(samplePanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
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

    private void enableButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_enableButtonActionPerformed
        activate();
    }//GEN-LAST:event_enableButtonActionPerformed

    private void itListMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_itListMouseClicked
        try {
            int roIdx = itList.getSelectedIndex();
                if (roIdx >= 0) {
                setRo(roIdx);
                enableControlPanel();
            }
        } catch (EasyGuiException ex) {
            setStatus(ex.getMessage(), true);
        }
    }//GEN-LAST:event_itListMouseClicked

    private void runRecButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_runRecButtonActionPerformed
        if (runRecButton.isSelected()) {
            if(!lcp.record()) lcp.record();
            lcp.getSequenceExtractor().clearBuffers();
            disableControllers();
            if (runRecDelaySlider.getValue() != 0) this.sync.setFreq(1);
            controller.setDelay(runRecDelay);
            delayTime = runRecDelay;
            controller.startMovie();
            setStatus("Recording images started");
        } else {
            if(lcp.record()) lcp.record();
            controller.stopMovie();
            setStatus("Recording images stopped");
            enableControllers();
        }
        registrationButton.setEnabled(true);
        runRecButton.setEnabled(true);
        paramButton.setEnabled(true);
    }//GEN-LAST:event_runRecButtonActionPerformed

    private void runRecDelaySliderStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_runRecDelaySliderStateChanged
        runRecDelay = runRecDelaySlider.getValue() * 50;
        runRecMsLabel.setText(runRecDelay + " ms");
        controller.setDelay(runRecDelay);
        delayTime = runRecDelay;
    }//GEN-LAST:event_runRecDelaySliderStateChanged

    private void paramButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_paramButtonActionPerformed
        lcp.doParamEstimation();
        setStatus("Parameter estimation");
    }//GEN-LAST:event_paramButtonActionPerformed

    private void runDelaySliderStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_runDelaySliderStateChanged
        runDelay = runDelaySlider.getValue() * 50;
        runMsLabel.setText(runDelay + " ms");
        controller.setDelay(runDelay);
        delayTime = runDelay;
    }//GEN-LAST:event_runDelaySliderStateChanged

    private void registrationButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_registrationButtonActionPerformed
        boolean registering = registrationButton.isSelected();
        regPanel.register(registering);
        if (registering) setStatus("Activated image registration");
        else setStatus("Deactivated image registration");
    }//GEN-LAST:event_registrationButtonActionPerformed

    private void photoButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_photoButtonActionPerformed
        lcp.getSequenceExtractor().clearBuffers();
        controller.takePhoto();
        delayTime = -2; // identifier delay time for photo mode
        setStatus("Captured a photo");
    }//GEN-LAST:event_photoButtonActionPerformed

    private void runButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_runButtonActionPerformed
        if (runButton.isSelected()) {
            lcp.getSequenceExtractor().clearBuffers();
            disableControllers();
            if (runDelaySlider.getValue() != 0) this.sync.setFreq(1);
            controller.setDelay(runDelay);
            delayTime = runDelay;
            controller.startMovie();
            setStatus("Capturing images started");
        } else {
            controller.stopMovie();
            setStatus("Capturing images stopped");
            enableControllers();
        }
        registrationButton.setEnabled(true);
        runButton.setEnabled(true);
        paramButton.setEnabled(true);
        enableRunRec();
    }//GEN-LAST:event_runButtonActionPerformed

    private void greenComboBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_greenComboBoxActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_greenComboBoxActionPerformed

    private void redComboBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_redComboBoxActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_redComboBoxActionPerformed

    private void addDyeButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_addDyeButtonActionPerformed
        addToDyeList();
    }//GEN-LAST:event_addDyeButtonActionPerformed

    private void dyeTextFieldKeyPressed(java.awt.event.KeyEvent evt) {//GEN-FIRST:event_dyeTextFieldKeyPressed
        if (evt.getKeyCode() == KeyEvent.VK_ENTER) {
            addToDyeList();
        }
    }//GEN-LAST:event_dyeTextFieldKeyPressed
    
    public String getDye(int channel) {
        List<String> dyes = new ArrayList<>(3);
        if (blueCheckBox.isSelected()) dyes.add(blueComboBox.getItemAt(blueComboBox.getSelectedIndex()));
        if (greenCheckBox.isSelected()) dyes.add(greenComboBox.getItemAt(greenComboBox.getSelectedIndex()));
        if (redCheckBox.isSelected()) dyes.add(redComboBox.getItemAt(redComboBox.getSelectedIndex()));
        try {
            return dyes.get(channel);
        } catch (IndexOutOfBoundsException ex) {
            return "-";
        }
    }
    
    public float getIlluminationPower(int channel) {
        List<Float> powers = new ArrayList<>(3);
        if (blueCheckBox.isSelected()) {
            try {
                powers.add(Float.parseFloat(blueTextField.getText()));
            } catch (NumberFormatException ex) {
                powers.add(new Float(0));
            }
        }
        if (greenCheckBox.isSelected()) {
            try {
                powers.add(Float.parseFloat(greenTextField.getText()));
            } catch (NumberFormatException ex) {
                powers.add(new Float(0));
            }
        }
        if (redCheckBox.isSelected()) {
            try {
                powers.add(Float.parseFloat(redTextField.getText()));
            } catch (NumberFormatException ex) {
                powers.add(new Float(0));
            }
        }
        try {
            return powers.get(channel);
        } catch (IndexOutOfBoundsException ex) {
            return 0;
        }
    }
    
    public int getIlluminationTime() {
        return illuminationTime;
    }
    
    public int getDelayTime() {
        return delayTime;
    }
    
    public int getSyncDelayTime() {
        return syncDelayTime;
    }
    
    public int getSyncFreq() {
        return syncFreq;
    }
    
    private void addToDyeList() {
        String dye = dyeTextField.getText() + "\n";
        if (dye.length() > 1) {
            System.out.println(dye.length());
            try {
                Files.write(Paths.get(dyeFile), dye.getBytes(), StandardOpenOption.APPEND);
            } catch (IOException ex) {
                throw new RuntimeException(ex);
            }
            dyeTextField.setText("");
            updateDyeBoxes();
        }
    }
    
    private void updateDyeBoxes() {
        blueComboBox.removeAllItems();
        blueComboBox.addItem("-");
        greenComboBox.removeAllItems();
        greenComboBox.addItem("-");
        redComboBox.removeAllItems();
        redComboBox.addItem("-");
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(dyeFile));
            String line = br.readLine();
            while (line != null) {
                blueComboBox.addItem(line);
                greenComboBox.addItem(line);
                redComboBox.addItem(line);
                line = br.readLine();
            }
        } catch(IOException ex) {
            Tool.error(ex.toString());
        } finally {
            try {
                br.close();
            } catch (IOException ex) {
                Tool.error(ex.toString());
            }
        }
    }
    
    
    
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
        enableRun();
        runButton.setEnabled(true);
        enableRunRec();
        runRecButton.setEnabled(true);
        photoButton.setEnabled(true);
        paramButton.setEnabled(true);
    }
    
    /**
     * disables the control panel
     */
    private void disableControlPanel() {
        controlPanel.setEnabled(false);
        registrationButton.setEnabled(false);
        disableRun();
        runButton.setEnabled(false);
        disableRunRec();
        runRecButton.setEnabled(false);
        photoButton.setEnabled(false);
        paramButton.setEnabled(false);
    }
    
    private void enableRunRec() {
        runRecDelayLabel.setEnabled(true);
        runRecDelaySlider.setEnabled(true);
        runRecMsLabel.setEnabled(true);
    }
    
    private void disableRunRec() {
        runRecDelayLabel.setEnabled(false);
        runRecDelaySlider.setEnabled(false);
        runRecMsLabel.setEnabled(false);
    }
        
    private void enableRun() {
        runDelayLabel.setEnabled(true);
        runDelaySlider.setEnabled(true);
        runMsLabel.setEnabled(true);
    }
    
    private void disableRun() {
        runDelayLabel.setEnabled(false);
        runDelaySlider.setEnabled(false);
        runMsLabel.setEnabled(false);
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
    private void setRo(int roIdx) throws EasyGuiException {
        for (Cam c : camGuis) {
            c.stopMovie();
        }
        if (roIdx >= 0) {
            RunningOrder ro = possibleRos.get(roIdx);
            controller.setRo(ro);
            for (Cam c : camGuis) {
                c.setRo(ro);
            }
            advanced.setRo(ro);
            sync.setRo(ro);
            for (Cam c : camGuis) {
                c.startMovie();
            }
            String illuminationUnit = ro.illuminationTime.substring(ro.illuminationTime.length()-2, ro.illuminationTime.length());
            illuminationTime = Integer.parseInt(ro.illuminationTime.substring(0, ro.illuminationTime.length()-2));
            if (illuminationUnit.equalsIgnoreCase("ms")) illuminationTime *= 1000;
            syncDelayTime = ro.syncDelay;
            syncFreq = ro.syncFreq;
            setStatus("Running order: " + ro.device + "_" + ro.name);
        }
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton addDyeButton;
    private javax.swing.JCheckBox blueCheckBox;
    private javax.swing.JComboBox<String> blueComboBox;
    private javax.swing.JLabel blueDyeLabel;
    private javax.swing.JLabel blueLabel;
    public javax.swing.JTextField blueTextField;
    private javax.swing.JPanel controlPanel;
    private javax.swing.JTextField dyeTextField;
    private javax.swing.JButton enableButton;
    private javax.swing.JCheckBox greenCheckBox;
    private javax.swing.JComboBox<String> greenComboBox;
    private javax.swing.JLabel greenDyeLabel;
    private javax.swing.JLabel greenLabel;
    public javax.swing.JTextField greenTextField;
    private javax.swing.JList<String> itList;
    private javax.swing.JPanel itPanel;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JPanel laserPanel;
    private javax.swing.JButton paramButton;
    private javax.swing.JButton photoButton;
    private javax.swing.JCheckBox redCheckBox;
    private javax.swing.JComboBox<String> redComboBox;
    private javax.swing.JLabel redDyeLabel;
    private javax.swing.JLabel redLabel;
    public javax.swing.JTextField redTextField;
    private javax.swing.JToggleButton registrationButton;
    private javax.swing.JToggleButton runButton;
    private javax.swing.JLabel runDelayLabel;
    private javax.swing.JSlider runDelaySlider;
    private javax.swing.JLabel runMsLabel;
    private javax.swing.JToggleButton runRecButton;
    private javax.swing.JLabel runRecDelayLabel;
    private javax.swing.JSlider runRecDelaySlider;
    private javax.swing.JLabel runRecMsLabel;
    private javax.swing.JPanel samplePanel;
    public javax.swing.JTextField sampleTextField;
    private javax.swing.JLabel statusLabel;
    private javax.swing.JPanel statusPanel;
    // End of variables declaration//GEN-END:variables
}
