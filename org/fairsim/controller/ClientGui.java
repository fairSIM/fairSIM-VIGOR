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
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.fairsim.utils.Conf;

/**
 *
 * @author m.lachetta
 */
public class ClientGui extends javax.swing.JPanel {

    Client client;
    private List<Component> slmControllers;
    private List<Component> arduinoControllers;
    private List<String> arduinoCommands;
    private boolean instructionDone;
    String serverAdress;
    int serverPort;

    /**
     * Creates new form SlmPanel
     *
     * @param cfg Configuration settings
     */
    public ClientGui(Conf.Folder cfg) {
        initComponents();

        try {
            serverAdress = cfg.getStr("SlmServerAdress").val();
        } catch (Conf.EntryNotFoundException ex) {
            serverAdress = "localhost";
        }
        try {
            serverPort = cfg.getInt("SlmServerPort").val();
        } catch (Conf.EntryNotFoundException ex) {
            serverPort = 32322;
        }

        Client.startClient(serverAdress, serverPort, this);

        initSlm();
        initArduino();
    }

    private void initSlm() {
        slmControllers = new ArrayList<Component>();
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

    private void initArduino() {
        arduinoCommands = new ArrayList<String>();
        arduinoCommands.add("m3n");
        arduinoCommands.add("m3o");
        arduinoCommands.add("m21r");
        arduinoCommands.add("m21g");
        arduinoCommands.add("m21b");
        arduinoCommands.add("m22r");
        arduinoCommands.add("m22g");
        arduinoCommands.add("m22b");
        arduinoCommands.add("m2Xr");
        arduinoCommands.add("m2Xg");
        arduinoCommands.add("m2Xb");
        arduinoCommands.add("m11r");
        arduinoCommands.add("m11g");
        arduinoCommands.add("m11b");
        arduinoCommands.add("m12r");
        arduinoCommands.add("m12g");
        arduinoCommands.add("m12b");
        arduinoCommands.add("m1Vr");
        arduinoCommands.add("m1Vg");
        arduinoCommands.add("m1Vb");
        arduinoCommands.add("m1Xr");
        arduinoCommands.add("m1Xg");
        arduinoCommands.add("m1Xb");
        for (String command : arduinoCommands) {
            arduinoComboBox.addItem(command);
        }

        arduinoControllers = new ArrayList<Component>();
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
     * Shows a new text-line in the text-field
     *
     * @param text String that shoulds be displayed in the text-field
     */
    void showText(String text) {
        textArea1.append(text + "\n");
    }

    /**
     * Sets lable on top of the gui
     *
     * @param adress server adress
     * @param port server port
     */
    void setConnectionLabel(String adress, int port) {
        serverLabel.setText("Server: " + adress + ":" + port);
    }

    /**
     * enables buttons of the GUI without connect-button
     */
    private void enableSlmControllers() {
        for (Component comp : slmControllers) {
            comp.setEnabled(true);
        }
    }

    /**
     * disables buttons of the GUI without connect-button
     */
    private void disableSlmControllers() {
        for (Component comp : slmControllers) {
            comp.setEnabled(false);
        }
    }

    private void enableArduinoControllers() {
        for (Component comp : arduinoControllers) {
            comp.setEnabled(true);
        }
    }

    private void disableArduinoControllers() {
        for (Component comp : arduinoControllers) {
            comp.setEnabled(false);
        }
    }

    void registerClient(Client client) {
        this.client = client;
        //showText("Gui: Client successfully registerd at the ControllerGui");
        slmConnectButton.setEnabled(true);
        arduinoConnectButton.setEnabled(true);
    }

    void unregisterClient() {
        //showText("Gui: Client unregisterd at the ControllerGui");

        disableSlmControllers();
        slmVersion.setText("SLM-API-Version -: ");
        slmTime.setText("Timestamp: -: ");
        slmType.setText("Activation type: -");
        slmDefault.setText("Default running order: -");
        slmSelect.setText("Selected running order: -");
        slmRepertoir.setText("Repertoir name: -");
        slmComboBox.removeAllItems();
        slmConnectButton.setEnabled(false);

        disableArduinoControllers();
    }

    void sendSlmInstruction(String command) {
        sendInstruction("slm->" + command);
    }

    void sendArduinoInstruction(String command) {
        sendInstruction("arduino->" + command);
    }

    /**
     * Sends a command for the server to the client
     *
     * @param command command for the server
     */
    void sendInstruction(String command) {
        Instruction slmInstruction = new Instruction(command);
        slmInstruction.lock.lock();
        try {
            //disableSlmControllers();
            client.instructions.add(slmInstruction);
            slmInstruction.condition.await(5, TimeUnit.SECONDS);
        } catch (InterruptedException ex) {
            showText("Gui: Error: Instruction timed out: " + ex.toString());
        } finally {
            slmInstruction.lock.unlock();
            //enableSlmControllers();
            try {
                if (client.output.startsWith("Slm: Error: ")) {
                    handleSlmError(client.output);
                    instructionDone = false;
                } else if (client.output.startsWith("Arduino: Error: ")) {
                    handleArduinoError(client.output);
                    instructionDone = false;
                } else {
                    instructionDone = true;
                }
            } catch (NullPointerException ex) {
                showText("Gui: Error: " + ex.toString());
                disableSlmControllers();
                slmConnectButton.setEnabled(false);
                disableArduinoControllers();
                arduinoConnectButton.setEnabled(false);
                instructionDone = false;
            }
        }
    }

    /**
     * handles error-massages from the server
     *
     * @param error Error-Massage from the server
     */
    private void handleSlmError(String error) {
        int code = Integer.parseInt(error.split("  ;  ")[1].split(": ")[1]);
        if (code == 12) {
            disconnectSLM();
            showText("Gui: Reconnect to the SLM is necessary");
        } else if (code == 7) {
            disableSlmControllers();
            slmConnectButton.setEnabled(true);
            showText("Gui: No connection to the SLM");
        } else if (code == 8) {
            disconnectSLM();
            connectSLM();
        }
    }

    private void handleArduinoError(String error) {
        if (error.contains("Could not find COM port")) {
            disableArduinoControllers();
            arduinoConnectButton.setEnabled(true);
            showText("Gui: No connection to the arduino");
        } else if (error.contains("IOException")) {
            disconnectArduino();
            showText("Gui: Reconnect to the arduino is necessary");
            connectArduino();
        } else if (error.contains("PortInUseException")) {
            disableArduinoControllers();
            arduinoConnectButton.setEnabled(true);
            showText("Gui: Arduino connection already in use");
        }
    }

    /**
     * refreshes the informations shown in the GUI
     */
    void refreshSlmInfo() {
        try {
            sendSlmInstruction("rolist");
            if (instructionDone) {
                slmComboBox.removeAllItems();
                for (int i = 0; i < client.roList.length; i++) {
                    slmComboBox.addItem("[" + i + "]    " + client.roList[i]);
                }
            }
            sendSlmInstruction("info");
            if (instructionDone) {
                slmVersion.setText("SLM-API-Version: " + client.info[0]);
                slmTime.setText("Timestamp: " + client.info[1]);
                slmType.setText("Activation type: " + client.info[2]);
                slmDefault.setText("Default running order: " + slmComboBox.getItemAt(Integer.parseInt(client.info[3])));
                slmSelect.setText("Selected running order: " + slmComboBox.getItemAt(Integer.parseInt(client.info[4])));
                slmRepertoir.setText("Repertoir name: " + client.info[5]);
            }
        } catch (NullPointerException ex) {
            System.err.println("Error while refreshing SLM-GUI");
            showText("Gui: Error: while refreshing SLM-GUI");
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

        slmPanel = new javax.swing.JPanel();
        slmVersion = new javax.swing.JLabel();
        slmTime = new javax.swing.JLabel();
        slmRepertoir = new javax.swing.JLabel();
        slmConnectButton = new javax.swing.JButton();
        slmComboBox = new javax.swing.JComboBox<>();
        slmDisconnectButton = new javax.swing.JButton();
        slmRebootButton = new javax.swing.JButton();
        slmDefault = new javax.swing.JLabel();
        slmSelect = new javax.swing.JLabel();
        slmType = new javax.swing.JLabel();
        slmActivateButton = new javax.swing.JButton();
        slmDeactivateButton = new javax.swing.JButton();
        slmSelectButton = new javax.swing.JButton();
        slmRefreshButton = new javax.swing.JButton();
        clientServerPanel = new javax.swing.JPanel();
        serverLabel = new javax.swing.JLabel();
        textArea1 = new java.awt.TextArea();
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

        slmPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("SLM-Controller"));
        slmPanel.setName(""); // NOI18N

        slmVersion.setText("SLM-API-Version: -");

        slmTime.setText("Timestamp: -");

        slmRepertoir.setText("Repertoir name: -");

        slmConnectButton.setText("Connect SLM");
        slmConnectButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                slmConnectButtonActionPerformed(evt);
            }
        });

        slmComboBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                slmComboBoxActionPerformed(evt);
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

        slmDefault.setText("Default running order: -");

        slmSelect.setText("Selected running order: -");

        slmType.setText("Activation type: -");

        slmActivateButton.setText("Aktivate");
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

        slmSelectButton.setText("Set Selected");
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
                .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addGroup(slmPanelLayout.createSequentialGroup()
                        .addComponent(slmConnectButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmDisconnectButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmRebootButton))
                    .addGroup(slmPanelLayout.createSequentialGroup()
                        .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addComponent(slmActivateButton, javax.swing.GroupLayout.PREFERRED_SIZE, 85, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(slmDeactivateButton, javax.swing.GroupLayout.Alignment.LEADING))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmComboBox, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(slmSelectButton)
                    .addComponent(slmRefreshButton))
                .addGap(0, 0, Short.MAX_VALUE))
            .addGroup(slmPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(slmRepertoir)
                    .addComponent(slmTime)
                    .addComponent(slmVersion))
                .addGap(114, 114, 114)
                .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(slmDefault)
                    .addComponent(slmSelect)
                    .addComponent(slmType))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
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
                    .addComponent(slmActivateButton)
                    .addComponent(slmComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(slmSelectButton))
                .addGap(3, 3, 3)
                .addComponent(slmDeactivateButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addGroup(slmPanelLayout.createSequentialGroup()
                        .addComponent(slmVersion)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmTime)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmRepertoir))
                    .addGroup(slmPanelLayout.createSequentialGroup()
                        .addComponent(slmDefault)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmSelect)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmType)))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

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
                .addComponent(textArea1, javax.swing.GroupLayout.DEFAULT_SIZE, 99, Short.MAX_VALUE))
        );

        arduinoPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Arduino-Controller"));

        arduinoComboBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                arduinoComboBoxActionPerformed(evt);
            }
        });

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
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
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
                        .addComponent(arduinoComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, 81, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoBreakTimeTextField, javax.swing.GroupLayout.PREFERRED_SIZE, 50, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(arduinoDelayLabel)
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );
        arduinoPanelLayout.setVerticalGroup(
            arduinoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(arduinoPanelLayout.createSequentialGroup()
                .addGroup(arduinoPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(arduinoConnectButton)
                    .addComponent(arduinoDisconnectButton)
                    .addComponent(arduinoBlueButton)
                    .addComponent(arduinoGreenButton)
                    .addComponent(arduinoRedButton))
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
                .addContainerGap(40, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(arduinoPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(slmPanel, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(clientServerPanel, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(slmPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(arduinoPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(clientServerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    private void slmDeactivateButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmDeactivateButtonActionPerformed
        sendSlmInstruction("deactivate");
    }//GEN-LAST:event_slmDeactivateButtonActionPerformed

    private void slmActivateButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmActivateButtonActionPerformed
        sendSlmInstruction("activate");
    }//GEN-LAST:event_slmActivateButtonActionPerformed

    private void slmComboBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmComboBoxActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_slmComboBoxActionPerformed

    private void slmSelectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmSelectButtonActionPerformed
        sendSlmInstruction(Integer.toString(slmComboBox.getSelectedIndex()));
        refreshSlmInfo();
    }//GEN-LAST:event_slmSelectButtonActionPerformed

    private void slmRefreshButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmRefreshButtonActionPerformed
        refreshSlmInfo();
    }//GEN-LAST:event_slmRefreshButtonActionPerformed

    private void slmRebootButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmRebootButtonActionPerformed
        sendSlmInstruction("reboot");
        if (instructionDone) {
            disableSlmControllers();
            Thread timer = new Thread(new Runnable() {
                @Override
                public void run() {
                    showText("Gui: Wait 20 Seconds until reboot finnished");
                    for (int time = 20; time > 0; time--) {
                        textArea1.append(time + "... ");
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException ex) {
                            Logger.getLogger(ClientGui.class.getName()).log(Level.SEVERE, null, ex);
                        }
                    }
                    textArea1.append("\n");
                    enableSlmControllers();
                }
            });
            timer.start();
        }
    }//GEN-LAST:event_slmRebootButtonActionPerformed

    private void slmDisconnectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmDisconnectButtonActionPerformed
        disconnectSLM();
    }//GEN-LAST:event_slmDisconnectButtonActionPerformed

    private void slmConnectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_slmConnectButtonActionPerformed
        connectSLM();
    }//GEN-LAST:event_slmConnectButtonActionPerformed

    private void arduinoConnectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoConnectButtonActionPerformed
        connectArduino();
    }//GEN-LAST:event_arduinoConnectButtonActionPerformed

    private void arduinoDisconnectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoDisconnectButtonActionPerformed
        disconnectArduino();
    }//GEN-LAST:event_arduinoDisconnectButtonActionPerformed

    private void arduinoStartButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoStartButtonActionPerformed
        try {
            startArduinoProgramm( (String) arduinoComboBox.getSelectedItem() + Integer.parseInt( arduinoBreakTimeTextField.getText() ));
        } catch (NumberFormatException ex) {
            startArduinoProgramm( (String) arduinoComboBox.getSelectedItem() );
        }
    }//GEN-LAST:event_arduinoStartButtonActionPerformed

    private void arduinoStopButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoStopButtonActionPerformed
        stopArduinoProgramm();
    }//GEN-LAST:event_arduinoStopButtonActionPerformed

    private void arduinoRedButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoRedButtonActionPerformed
        if ( arduinoRedButton.isSelected() ) {
            sendArduinoInstruction("R");
        } else {
            sendArduinoInstruction("r");
        }
    }//GEN-LAST:event_arduinoRedButtonActionPerformed

    private void arduinoGreenButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoGreenButtonActionPerformed
        if ( arduinoGreenButton.isSelected() ) {
            sendArduinoInstruction("G");
        } else {
            sendArduinoInstruction("g");
        }
    }//GEN-LAST:event_arduinoGreenButtonActionPerformed

    private void arduinoBlueButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoBlueButtonActionPerformed
        if ( arduinoBlueButton.isSelected() ) {
            sendArduinoInstruction("B");
        } else {
            sendArduinoInstruction("b");
        }
    }//GEN-LAST:event_arduinoBlueButtonActionPerformed

    private void arduinoPhotoButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoPhotoButtonActionPerformed
        sendArduinoInstruction( (String) arduinoComboBox.getSelectedItem() + "xx" );
        if (instructionDone) {
            setRGBButtonSelected(false);
        }
    }//GEN-LAST:event_arduinoPhotoButtonActionPerformed

    private void arduinoComboBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoComboBoxActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_arduinoComboBoxActionPerformed

    private void setRGBButtonSelected(boolean b) {
        arduinoRedButton.setSelected(b);
        arduinoGreenButton.setSelected(b);
        arduinoBlueButton.setSelected(b);
    }
    
    private void stopArduinoProgramm() {
        sendArduinoInstruction("x");
        if (instructionDone) {
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
    
    private void startArduinoProgramm(String command) {
        sendArduinoInstruction(command);
        if (instructionDone) {
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
     * Connects server and SLM
     */
    private void connectSLM() {
        sendSlmInstruction("connect");
        if (instructionDone) {
            refreshSlmInfo();
            enableSlmControllers();
            slmConnectButton.setEnabled(false);
        }
    }

    /**
     * Disconnects server and SLM
     */
    private void disconnectSLM() {
        sendSlmInstruction("disconnect");
        if (instructionDone) {
            disableSlmControllers();
            slmConnectButton.setEnabled(true);
        }
    }

    private void connectArduino() {
        sendArduinoInstruction("connect");
        if (instructionDone) {
            setRGBButtonSelected(false);
            arduinoConnectButton.setEnabled(false);
            final int waiting = 3;
            textArea1.append("Gui: Waiting for the arduino: ");
            Thread timer = new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        for (int i = waiting; i > 0; i--) {
                            textArea1.append(i + "... ");
                            Thread.sleep(1000);
                        }
                        textArea1.append("0 \n");
                    } catch (InterruptedException ex) {
                        showText("Gui: Error: Arduino sleep interrupted");
                    } finally {
                        enableArduinoControllers();
                        arduinoStopButton.setEnabled(false);
                    }
                }
                
            });
            timer.start();
        }
    }

    private void disconnectArduino() {
        sendArduinoInstruction("disconnect");
        if (instructionDone) {
            disableArduinoControllers();
            arduinoConnectButton.setEnabled(true);
        }
    }


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JToggleButton arduinoBlueButton;
    private javax.swing.JTextField arduinoBreakTimeTextField;
    private javax.swing.JComboBox<String> arduinoComboBox;
    private javax.swing.JButton arduinoConnectButton;
    private javax.swing.JLabel arduinoDelayLabel;
    private javax.swing.JButton arduinoDisconnectButton;
    private javax.swing.JToggleButton arduinoGreenButton;
    private javax.swing.JPanel arduinoPanel;
    private javax.swing.JButton arduinoPhotoButton;
    private javax.swing.JToggleButton arduinoRedButton;
    private javax.swing.JButton arduinoStartButton;
    private javax.swing.JButton arduinoStopButton;
    private javax.swing.JPanel clientServerPanel;
    private javax.swing.JLabel serverLabel;
    private javax.swing.JButton slmActivateButton;
    private javax.swing.JComboBox<String> slmComboBox;
    private javax.swing.JButton slmConnectButton;
    private javax.swing.JButton slmDeactivateButton;
    private javax.swing.JLabel slmDefault;
    private javax.swing.JButton slmDisconnectButton;
    private javax.swing.JPanel slmPanel;
    private javax.swing.JButton slmRebootButton;
    private javax.swing.JButton slmRefreshButton;
    private javax.swing.JLabel slmRepertoir;
    private javax.swing.JLabel slmSelect;
    private javax.swing.JButton slmSelectButton;
    private javax.swing.JLabel slmTime;
    private javax.swing.JLabel slmType;
    private javax.swing.JLabel slmVersion;
    private java.awt.TextArea textArea1;
    // End of variables declaration//GEN-END:variables
}
