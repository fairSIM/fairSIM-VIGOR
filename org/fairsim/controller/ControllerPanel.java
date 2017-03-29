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

import java.awt.Component;
import java.util.ArrayList;
import java.util.List;
import org.fairsim.livemode.SimSequenceExtractor;

/**
 * gui for a controller client
 * @author m.lachetta
 */
public class ControllerPanel extends javax.swing.JPanel implements AbstractClient.ClientGui, EasyGui.Ctrl {
    private AdvancedGui motherGui;
    ControllerClient controllerClient;
    private List<Component> slmControllers, arduinoControllers;
    private boolean controllerInstructionDone;
    private boolean slmConnected = false, slmReboot = false, arduinoConnected = false;
    SimSequenceExtractor seqDetection;
    
    /**
     * Creates new form ArduinoPanel
     */
    public ControllerPanel() {
        initComponents();
    }
    
    /**
     * enables this panel, called from the advanced gui
     * @param motherGui the advanced gui
     * @param adress server adress
     * @param port server port
     * @param seqDetection the sequence extractor of fairSIM
     */
    void enablePanel(AdvancedGui motherGui, String adress, int port, SimSequenceExtractor seqDetection) {
        initSlm();
        initArduino();
        
        this.motherGui = motherGui;
        this.seqDetection = seqDetection;
        
        controllerClient = new ControllerClient(adress, port, this);
        controllerClient.start();
    }
    
    /**
     * initializes the GUI for the SLM
     */
    private void initSlm() {
        slmControllers = new ArrayList<>();
        slmControllers.add(slmComboBox);
        slmControllers.add(slmDeactivateButton);
        slmControllers.add(slmActivateButton);
        slmControllers.add(slmSelectButton);
        slmControllers.add(slmDisconnectButton);
        slmControllers.add(slmRefreshButton);
        slmControllers.add(slmRebootButton);
        disableSlmControllers();
        slmConnectButton.setEnabled(false);
    }

    /**
     * initializes the GUI for the arduino
     */
    private void initArduino() {
        arduinoControllers = new ArrayList<>();
        arduinoControllers.add(arduinoComboBox);
        arduinoControllers.add(arduinoDisconnectButton);
        arduinoControllers.add(arduinoStartButton);
        arduinoControllers.add(arduinoStopButton);
        arduinoControllers.add(arduinoRedButton);
        arduinoControllers.add(arduinoGreenButton);
        arduinoControllers.add(arduinoBlueButton);
        arduinoControllers.add(arduinoBreakTimeTextField);
        arduinoControllers.add(arduinoPhotoButton);
        disableArduinoControllers();
        arduinoConnectButton.setEnabled(false);
    }
    
    /**
     * enables buttons of the SLM without connect-button
     */
    private void enableSlmControllers() {
        for (Component comp : slmControllers) {
            comp.setEnabled(true);
        }
    }

    /**
     * disables buttons of the SLM without connect-button
     */
    private void disableSlmControllers() {
        for (Component comp : slmControllers) {
            comp.setEnabled(false);
        }
    }

    /**
     * enables buttons of the arduino without connect-button
     */
    private void enableArduinoControllers() {
        for (Component comp : arduinoControllers) {
            comp.setEnabled(true);
        }
    }

    /**
     * disables buttons of the arduino without connect-button
     */
    private void disableArduinoControllers() {
        for (Component comp : arduinoControllers) {
            comp.setEnabled(false);
        }
    }    

    /**
     * Sends a command to the SLM
     *
     * @param command
     */
    private void sendSlmInstruction(String command) {
        sendControllerInstruction("slm->" + command);
    }

    /**
     * Sends a command to the arduino
     *
     * @param command
     */
    private void sendArduinoInstruction(String command) {
        sendControllerInstruction("arduino->" + command);
    }

    /**
     * Sends a command for the server to the client
     *
     * @param command command for the server
     */
    private void sendControllerInstruction(String command) {
        controllerInstructionDone = true;
        controllerClient.addInstruction(command);
    }

    /**
     * handles error-massages from the SLM
     *
     * @param error Error-Massage from the SLM
     */
    void handleSlmError(String error) {
        controllerInstructionDone = false;
        int code = Integer.parseInt(error.split("  ;  ")[1].split(": ")[1]);
        if (code == 12) {
            slmDissconnect();
            showText("Gui: Reconnect to the SLM is necessary");
        } else if (code == 7) {
            disableSlmControllers();
            slmConnectButton.setEnabled(true);
            slmConnected = false;
            showText("Gui: No connection to the SLM");
        } else if (code == 8) {
            slmDissconnect();
            slmConnect();
        }
    }

    /**
     * handles an error-massage from the Arduino
     *
     * @param error
     */
    void handleArduinoError(String error) {
        controllerInstructionDone = false;
        if (error.contains("Gui: Could not find COM port")) {
            disableArduinoControllers();
            arduinoConnectButton.setEnabled(true);
            showText("Gui: No connection to the arduino");
        } else if (error.contains("IOException")) {
            arduinoDisconnect();
            showText("Gui: Reconnect to the arduino is necessary");
            arduinoConnect();
        } else if (error.contains("PortInUseException")) {
            disableArduinoControllers();
            arduinoConnectButton.setEnabled(true);
            showText("Gui: Arduino connection already in use");
        }
    }

    /**
     * refreshes the informations shown in the GUI
     */
    void slmRefresh() {
        try {
            sendSlmInstruction("info");
            if (controllerInstructionDone) {
                //slmVersion.setText("SLM-API-Version: " + client.info[0]);
                //slmTime.setText("Timestamp: " + client.info[1]);
                //slmType.setText("Activation type: " + client.info[2]);
                //slmDefault.setText("Default running order: " + slmComboBox.getItemAt(Integer.parseInt(client.info[3])));
                slmSelect.setText("Selected running order: " + slmComboBox.getItemAt(Integer.parseInt(controllerClient.deviceInfo[4])));
                //slmRepertoir.setText("Repertoir name: " + client.info[5]);
            }
        } catch (NullPointerException ex) {
            System.err.println("Error while refreshing SLM-GUI");
            showText("[fairSIM] Error while refreshing SLM-GUI");
        }
    }
    
    /**
     * sets the red/green/blue buttons to selected/unselected
     * @param b true/false = selected/unselected
     */
    private void setRGBButtonSelected(boolean b) {
        arduinoRedButton.setSelected(b);
        arduinoGreenButton.setSelected(b);
        arduinoBlueButton.setSelected(b);
    }

    /**
     * stops the the program on the arduino
     */
    void arduinoStop() {
        sendArduinoInstruction("x");
        if (controllerInstructionDone) {
            arduinoStartButton.setEnabled(true);
            arduinoRedButton.setEnabled(true);
            arduinoGreenButton.setEnabled(true);
            arduinoBlueButton.setEnabled(true);
            arduinoComboBox.setEnabled(true);
            arduinoBreakTimeTextField.setEnabled(true);
            arduinoPhotoButton.setEnabled(true);
            arduinoStopButton.setEnabled(false);
        }
    }

    /**
     * starts a arduino program
     * @param command command for the arduino
     */
    private void startArduinoProgramm(String command) {
        seqDetection.resetChannelBufferThreads();
        sendArduinoInstruction(command);
        if (controllerInstructionDone) {
            arduinoStartButton.setEnabled(false);
            arduinoRedButton.setEnabled(false);
            arduinoGreenButton.setEnabled(false);
            arduinoBlueButton.setEnabled(false);
            arduinoComboBox.setEnabled(false);
            arduinoBreakTimeTextField.setEnabled(false);
            arduinoPhotoButton.setEnabled(false);
            arduinoStopButton.setEnabled(true);
            setRGBButtonSelected(false);
        }
    }

    /**
     * Connects server and SLM and updates the running orders
     */
    void slmConnect() {
        // connecting
        sendSlmInstruction("connect");
        if (controllerInstructionDone) {
            // updating running orders
            sendSlmInstruction("rolist");
            if (controllerInstructionDone) {
                slmComboBox.removeAllItems();
                for (int i = 0; i < controllerClient.deviceList.length; i++) {
                    slmComboBox.addItem("[" + i + "]    " + controllerClient.deviceList[i]);
                }
            }
            slmRefresh();
            enableSlmControllers();
            slmConnectButton.setEnabled(false);
            slmConnected = true;
        }
    }

    /**
     * Disconnects server and SLM
     */
    void slmDissconnect() {
        sendSlmInstruction("disconnect");
        if (controllerInstructionDone) {
            disableSlmControllers();
            slmConnectButton.setEnabled(true);
            slmConnected = false;
        }
    }

    /**
     * connects server and arudino
     */
    void arduinoConnect() {
        sendArduinoInstruction("connect");
        if (controllerInstructionDone) {
            setRGBButtonSelected(false);
            arduinoConnectButton.setEnabled(false);
            int waiting = 3;
            showText("Gui: Waiting for the arduino... (" + waiting + "seconds)");
            try {
                for (int i = waiting; i > 0; i--) {
                    Thread.sleep(1000);
                }
            } catch (InterruptedException ex) {
                showText("Gui: Error: Arduino sleep interrupted");
            } finally {
                updateArduinoRos();
                enableArduinoControllers();
                arduinoStopButton.setEnabled(false);
                arduinoConnected = true;
            }
        }
    }
    
    /**
     * updates running orders from the arduino
     */
    private void updateArduinoRos() {
        sendArduinoInstruction("rolist");
            if (controllerInstructionDone) {
                arduinoComboBox.removeAllItems();
                for (int i = 0; i < controllerClient.arduinoRos.length; i++) {
                    arduinoComboBox.addItem("[" + i + "]    " + controllerClient.arduinoRos[i].name);
                }
            }
    }

    /**
     * disconnects server and arduino
     */
    void arduinoDisconnect() {
        sendArduinoInstruction("disconnect");
        if (controllerInstructionDone) {
            disableArduinoControllers();
            arduinoConnectButton.setEnabled(true);
            arduinoConnected = false;
        }
    }
    
    /**
     * starts the selected movie running order of the arduino from this gui
     */
    void arduinoStart() {
        int breakTime = 0;
        try {
            breakTime = Integer.parseInt(arduinoBreakTimeTextField.getText());
        } catch (NumberFormatException ex) {}
        if (breakTime < 0) breakTime = 0;
        startArduinoProgramm("movie;" + arduinoComboBox.getSelectedIndex() + ";" + breakTime);
    }
    
    /**
     * starts the selected photo running order of the arduino from this gui
     */
    void arduinoPhoto() {
        seqDetection.resetChannelBufferThreads();
        sendArduinoInstruction("photo;" + arduinoComboBox.getSelectedIndex());
        if (controllerInstructionDone) {
            setRGBButtonSelected(false);
        }
    }
    
    /**
     * sets the selected running order of the slm
     */
    void slmSetSelected() {
        sendSlmInstruction(Integer.toString(slmComboBox.getSelectedIndex()));
        slmRefresh();
    }
    
    /**
     * deactivate the running oder on the slm
     */
    void slmDeactivate() {
        sendSlmInstruction("deactivate");
    }
    
    /**
     * activates the running order on the slm
     */
    void slmActivate() {
        sendSlmInstruction("activate");
    }
    
    /**
     * 
     * @return running orders of the arduino
     */
    public ControllerClient.ArduinoRunningOrder[] getArduinoRos() {
        return controllerClient.arduinoRos;
    }
    
    /**
     * 
     * @return running orders of the slm
     */
    public String[] getDeviceRos() {
        return controllerClient.deviceList;
    }
    
    @Override
    public void showText(String text) {
        motherGui.showText(text);
    }
    
    @Override
    public void registerClient() {
        slmConnectButton.setEnabled(true);
        arduinoConnectButton.setEnabled(true);
    }
    
    @Override
    public void unregisterClient() {
        disableSlmControllers();
        slmSelect.setText("Selected running order: -");
        slmComboBox.removeAllItems();
        slmConnectButton.setEnabled(false);
        disableArduinoControllers();
        slmConnected = false;
        arduinoConnected = false;
    }

    @Override
    public void handleError(String answer) {
        if (answer.startsWith("Slm: Error: ")) {
            handleSlmError(answer);
        } else if (answer.startsWith("Arduino: Error: ")) {
            handleArduinoError(answer);
        }
    }

    @Override
    public void interruptInstruction() {
        controllerInstructionDone = false;
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        slmPanel = new javax.swing.JPanel();
        slmConnectButton = new javax.swing.JButton();
        slmComboBox = new javax.swing.JComboBox<>();
        slmDisconnectButton = new javax.swing.JButton();
        slmRebootButton = new javax.swing.JButton();
        slmSelect = new javax.swing.JLabel();
        slmActivateButton = new javax.swing.JButton();
        slmDeactivateButton = new javax.swing.JButton();
        slmSelectButton = new javax.swing.JButton();
        slmRefreshButton = new javax.swing.JButton();
        arduinoPanel = new javax.swing.JPanel();
        arduinoComboBox = new javax.swing.JComboBox<>();
        arduinoConnectButton = new javax.swing.JButton();
        arduinoDisconnectButton = new javax.swing.JButton();
        arduinoStartButton = new javax.swing.JButton();
        arduinoStopButton = new javax.swing.JButton();
        arduinoRedButton = new javax.swing.JToggleButton();
        arduinoGreenButton = new javax.swing.JToggleButton();
        arduinoBlueButton = new javax.swing.JToggleButton();
        arduinoBreakTimeTextField = new javax.swing.JTextField();
        arduinoDelayLabel = new javax.swing.JLabel();
        arduinoPhotoButton = new javax.swing.JButton();
        arduinoLasersLabel = new javax.swing.JLabel();

        slmPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("SLM-Controller"));
        slmPanel.setName(""); // NOI18N

        slmConnectButton.setText("Connect SLM");
        slmConnectButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                slmConnectButtonActionPerformed(evt);
            }
        });

        slmDisconnectButton.setText("Disconnect SLM");
        slmDisconnectButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                slmDisconnectButtonActionPerformed(evt);
            }
        });

        slmRebootButton.setText("(Reboot SLM)");
        slmRebootButton.setToolTipText("Maybe it works, maybe not");
        slmRebootButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                slmRebootButtonActionPerformed(evt);
            }
        });

        slmSelect.setText("Selected running order: -");

        slmActivateButton.setText("Activate");
        slmActivateButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                slmActivateButtonActionPerformed(evt);
            }
        });

        slmDeactivateButton.setText("Deactivate");
        slmDeactivateButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                slmDeactivateButtonActionPerformed(evt);
            }
        });

        slmSelectButton.setText("Selected");
        slmSelectButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                slmSelectButtonActionPerformed(evt);
            }
        });

        slmRefreshButton.setText("Refresh GUI");
        slmRefreshButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                slmRefreshButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout slmPanelLayout = new javax.swing.GroupLayout(slmPanel);
        slmPanel.setLayout(slmPanelLayout);
        slmPanelLayout.setHorizontalGroup(
            slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(slmPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(slmPanelLayout.createSequentialGroup()
                        .addComponent(slmConnectButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmDisconnectButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmRebootButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmRefreshButton))
                    .addComponent(slmSelect)
                    .addGroup(slmPanelLayout.createSequentialGroup()
                        .addComponent(slmSelectButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmComboBox, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmActivateButton, javax.swing.GroupLayout.PREFERRED_SIZE, 85, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmDeactivateButton)
                        .addGap(47, 47, 47))))
        );
        slmPanelLayout.setVerticalGroup(
            slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, slmPanelLayout.createSequentialGroup()
                .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(slmConnectButton)
                    .addComponent(slmDisconnectButton)
                    .addComponent(slmRebootButton)
                    .addComponent(slmRefreshButton))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(slmComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(slmSelectButton)
                    .addComponent(slmActivateButton)
                    .addComponent(slmDeactivateButton))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(slmSelect)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        arduinoPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Arduino-Controller"));

        arduinoConnectButton.setText("Connect Arduino");
        arduinoConnectButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                arduinoConnectButtonActionPerformed(evt);
            }
        });

        arduinoDisconnectButton.setText("Disconnect Arduino");
        arduinoDisconnectButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                arduinoDisconnectButtonActionPerformed(evt);
            }
        });

        arduinoStartButton.setText("Start Program");
        arduinoStartButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                arduinoStartButtonActionPerformed(evt);
            }
        });

        arduinoStopButton.setText("Stop Program");
        arduinoStopButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                arduinoStopButtonActionPerformed(evt);
            }
        });

        arduinoRedButton.setForeground(new java.awt.Color(255, 0, 0));
        arduinoRedButton.setText("Red");
        arduinoRedButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                arduinoRedButtonActionPerformed(evt);
            }
        });

        arduinoGreenButton.setForeground(new java.awt.Color(0, 255, 0));
        arduinoGreenButton.setText("Green");
        arduinoGreenButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                arduinoGreenButtonActionPerformed(evt);
            }
        });

        arduinoBlueButton.setForeground(new java.awt.Color(0, 0, 255));
        arduinoBlueButton.setText("Blue");
        arduinoBlueButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                arduinoBlueButtonActionPerformed(evt);
            }
        });

        arduinoBreakTimeTextField.setText("0");

        arduinoDelayLabel.setText("delay (ms)");

        arduinoPhotoButton.setText("Photo");
        arduinoPhotoButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                arduinoPhotoButtonActionPerformed(evt);
            }
        });

        arduinoLasersLabel.setText("Lasers:");

        javax.swing.GroupLayout arduinoPanelLayout = new javax.swing.GroupLayout(arduinoPanel);
        arduinoPanel.setLayout(arduinoPanelLayout);
        arduinoPanelLayout.setHorizontalGroup(
            arduinoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(arduinoPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(arduinoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(arduinoPanelLayout.createSequentialGroup()
                        .addComponent(arduinoConnectButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoDisconnectButton)
                        .addGap(18, 18, 18)
                        .addComponent(arduinoLasersLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoRedButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoGreenButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoBlueButton))
                    .addGroup(arduinoPanelLayout.createSequentialGroup()
                        .addComponent(arduinoStartButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoStopButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoPhotoButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoComboBox, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoBreakTimeTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 50, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoDelayLabel)))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        arduinoPanelLayout.setVerticalGroup(
            arduinoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(arduinoPanelLayout.createSequentialGroup()
                .addGroup(arduinoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(arduinoConnectButton)
                    .addComponent(arduinoDisconnectButton)
                    .addComponent(arduinoBlueButton)
                    .addComponent(arduinoGreenButton)
                    .addComponent(arduinoRedButton)
                    .addComponent(arduinoLasersLabel))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(arduinoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(arduinoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                        .addComponent(arduinoComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addComponent(arduinoBreakTimeTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addComponent(arduinoDelayLabel)
                        .addComponent(arduinoPhotoButton))
                    .addGroup(arduinoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                        .addComponent(arduinoStopButton)
                        .addComponent(arduinoStartButton)))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(slmPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addComponent(arduinoPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addComponent(slmPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(arduinoPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents

    private void arduinoConnectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoConnectButtonActionPerformed
        arduinoConnect();
    }//GEN-LAST:event_arduinoConnectButtonActionPerformed

    private void arduinoDisconnectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoDisconnectButtonActionPerformed
        arduinoDisconnect();
    }//GEN-LAST:event_arduinoDisconnectButtonActionPerformed

    private void arduinoStartButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoStartButtonActionPerformed
        arduinoStart();
    }//GEN-LAST:event_arduinoStartButtonActionPerformed

    private void arduinoStopButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoStopButtonActionPerformed
        arduinoStop();
    }//GEN-LAST:event_arduinoStopButtonActionPerformed

    private void arduinoRedButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoRedButtonActionPerformed
        if (arduinoRedButton.isSelected()) {
            sendArduinoInstruction("R");
        } else {
            sendArduinoInstruction("r");
        }
    }//GEN-LAST:event_arduinoRedButtonActionPerformed

    private void arduinoGreenButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoGreenButtonActionPerformed
        if (arduinoGreenButton.isSelected()) {
            sendArduinoInstruction("G");
        } else {
            sendArduinoInstruction("g");
        }
    }//GEN-LAST:event_arduinoGreenButtonActionPerformed

    private void arduinoBlueButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoBlueButtonActionPerformed
        if (arduinoBlueButton.isSelected()) {
            sendArduinoInstruction("B");
        } else {
            sendArduinoInstruction("b");
        }
    }//GEN-LAST:event_arduinoBlueButtonActionPerformed

    private void arduinoPhotoButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoPhotoButtonActionPerformed
        arduinoPhoto();
    }//GEN-LAST:event_arduinoPhotoButtonActionPerformed

    private void slmRefreshButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmRefreshButtonActionPerformed
        slmRefresh();
    }//GEN-LAST:event_slmRefreshButtonActionPerformed

    private void slmSelectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmSelectButtonActionPerformed
        slmSetSelected();
    }//GEN-LAST:event_slmSelectButtonActionPerformed

    private void slmDeactivateButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmDeactivateButtonActionPerformed
        slmDeactivate();
    }//GEN-LAST:event_slmDeactivateButtonActionPerformed

    private void slmActivateButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmActivateButtonActionPerformed
        slmActivate();
    }//GEN-LAST:event_slmActivateButtonActionPerformed

    private void slmRebootButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmRebootButtonActionPerformed
        sendSlmInstruction("reboot");
        if (controllerInstructionDone) {
            slmReboot = true;
            disableSlmControllers();
            Thread timer = new Thread(new Runnable() {
                @Override
                public void run() {
                    int time = 20;
                    showText("Gui: Wait " + time + " Seconds until reboot finnished...");
                    for (int i = 0; i < time; i++) {
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException ex) {
                            throw new RuntimeException("Woken up while rebooting SLM");
                        }
                    }
                    showText("Gui: SLM rebooted");
                    enableSlmControllers();
                    slmReboot = false;
                }
            });
            timer.start();
        }
    }//GEN-LAST:event_slmRebootButtonActionPerformed

    private void slmDisconnectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmDisconnectButtonActionPerformed
        slmDissconnect();
    }//GEN-LAST:event_slmDisconnectButtonActionPerformed

    private void slmConnectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmConnectButtonActionPerformed
        slmConnect();
    }//GEN-LAST:event_slmConnectButtonActionPerformed


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JToggleButton arduinoBlueButton;
    javax.swing.JTextField arduinoBreakTimeTextField;
    javax.swing.JComboBox<String> arduinoComboBox;
    private javax.swing.JButton arduinoConnectButton;
    private javax.swing.JLabel arduinoDelayLabel;
    private javax.swing.JButton arduinoDisconnectButton;
    private javax.swing.JToggleButton arduinoGreenButton;
    private javax.swing.JLabel arduinoLasersLabel;
    private javax.swing.JPanel arduinoPanel;
    private javax.swing.JButton arduinoPhotoButton;
    private javax.swing.JToggleButton arduinoRedButton;
    private javax.swing.JButton arduinoStartButton;
    private javax.swing.JButton arduinoStopButton;
    private javax.swing.JButton slmActivateButton;
    javax.swing.JComboBox<String> slmComboBox;
    private javax.swing.JButton slmConnectButton;
    private javax.swing.JButton slmDeactivateButton;
    private javax.swing.JButton slmDisconnectButton;
    private javax.swing.JPanel slmPanel;
    private javax.swing.JButton slmRebootButton;
    private javax.swing.JButton slmRefreshButton;
    private javax.swing.JLabel slmSelect;
    private javax.swing.JButton slmSelectButton;
    // End of variables declaration//GEN-END:variables

    @Override
    public void enableEasy() throws EasyGui.EasyGuiException{
        if (!slmConnected) {
            slmConnect();
            if (!slmConnected) throw new EasyGui.EasyGuiException("Controller: No connection to the slm");
        }
        if (!arduinoConnected) {
            arduinoConnect();
            if (!arduinoConnected) throw new EasyGui.EasyGuiException("Controller: No connection to the arduino");
        }
    }
    
    @Override
    public void setDelay(int delay) {
        arduinoBreakTimeTextField.setText(String.valueOf(delay));
    }
    
    @Override
    public void setRo(EasyGui.RunningOrder ro) throws EasyGui.EasyGuiException {
        arduinoComboBox.setSelectedIndex(ro.arduinoRo);
        if (ro.device.equals("slm")) slmComboBox.setSelectedIndex(ro.deviceRo);
        else throw new EasyGui.EasyGuiException("Controller: Not supported device: " + ro.device);
        slmDeactivate();
        slmSetSelected();
        slmDeactivate();
        slmActivate();
        slmDeactivate();
        slmDeactivate();
        slmDeactivate();
        slmDeactivate();
        slmDeactivate();
        slmActivate();
        slmActivate();
        slmActivate();
        slmActivate();
        slmActivate();
    }
    
    @Override
    public void takePhoto() {
        this.arduinoPhoto();
    }

    @Override
    public void startMovie() {
        this.arduinoStart();
    }

    @Override
    public void stopMovie() {
        this.arduinoStop();
    }
}
