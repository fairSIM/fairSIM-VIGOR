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
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.zip.DataFormatException;
import org.fairsim.livemode.ReconstructionRunner;
import org.fairsim.livemode.SimSequenceExtractor;
import org.fairsim.registration.RegFileCreatorGui;
import org.fairsim.registration.Registration;
import org.fairsim.utils.Conf;

/**
 *
 * @author m.lachetta
 */
public class ControllerClientGui extends javax.swing.JPanel implements ClientGui {

    ControllerClient controllerClient;
    private List<Component> slmControllers;
    private List<Component> arduinoControllers;
    private List<String> arduinoCommands;
    private boolean instructionDone;
    String serverAdress;
    int serverPort;
    String regFolder;
    String[] channelNames;
    SimSequenceExtractor seqDetection;
    ReconstructionRunner recRunner;

    /**
     * Creates the GUI for the Controller
     *
     * @param cfg Configuration settings
     * @param channelNames Camera Channels
     * @param seqDetection The Sim-Sequence-Extractor
     */
    public ControllerClientGui(Conf.Folder cfg, String[] channelNames, SimSequenceExtractor seqDetection, ReconstructionRunner recRunner) {
        this.seqDetection = seqDetection;
        this.recRunner = recRunner;

        initComponents();

        try {
            serverAdress = cfg.getStr("SlmServerAdress").val();
        } catch (Conf.EntryNotFoundException ex) {
            serverAdress = "localhost";
            System.err.println("[fairSIM]: Error: no SlmServerAdress found. SLMServerAdress set to 'localhost'");
        }
        try {
            serverPort = cfg.getInt("SlmServerPort").val();
        } catch (Conf.EntryNotFoundException ex) {
            serverPort = 32322;
            System.err.println("[fairSIM]: Error: no SlmServerPort found. SLMServerPort set to '32322'");
        }

        this.channelNames = channelNames;

        //ControllerClient.startClient(serverAdress, serverPort, this);
        controllerClient = new ControllerClient(serverAdress, serverPort, this);
        controllerClient.start();
        setConnectionLabel(serverAdress, serverPort);

        initSlm();
        initArduino();
        initSync();
        initReg(cfg);
    }

    /**
     * Initialises the GUI for the SLM
     */
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

    /**
     * Initialises the GUI for the arduino
     */
    private void initArduino() {
        arduinoCommands = new ArrayList<String>();
        arduinoCommands.add("m3f");
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
            System.err.println(ex.getMessage());
        } catch (ClassNotFoundException ex) {
            regCreatorButton.setEnabled(false);
            System.err.println("[fairSIM]: Error: jar files for bunwarpj and/or imagej are missing. Deaktived registration-creator.");
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
    void setConnectionLabel(String adress, int port) {
        serverLabel.setText("Server: " + adress + ":" + port);
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
     * registers the client at the GUI
     *
     * @param client
     */
    public void registerClient(AbstractClient client) {
        if (client instanceof ControllerClient) {
            controllerClient = (ControllerClient) client;
            slmConnectButton.setEnabled(true);
            arduinoConnectButton.setEnabled(true);
        }
    }
    
    /*
    void registerClient(ControllerClient client) {
        controllerClient = client;
        slmConnectButton.setEnabled(true);
        arduinoConnectButton.setEnabled(true);
    }
    */
    /**
     * unregisters the client at the GUI
     */
    public void unregisterClient(AbstractClient client) {
        if (client instanceof ControllerClient) {
            disableSlmControllers();
            //slmVersion.setText("SLM-API-Version -: ");
            //slmTime.setText("Timestamp: -: ");
            //slmType.setText("Activation type: -");
            //slmDefault.setText("Default running order: -");
            slmSelect.setText("Selected running order: -");
            //slmRepertoir.setText("Repertoir name: -");
            slmComboBox.removeAllItems();
            slmConnectButton.setEnabled(false);

            disableArduinoControllers();
        }
    }

    /**
     * Sends a command to the SLM
     *
     * @param command
     */
    void sendSlmInstruction(String command) {
        sendInstruction("slm->" + command);
    }

    /**
     * Sends a command to the arduino
     *
     * @param command
     */
    void sendArduinoInstruction(String command) {
        sendInstruction("arduino->" + command);
    }

    /**
     * Sends a command for the server to the client
     *
     * @param command command for the server
     */
    void sendInstruction(String command) {
        Instruction instruction = new Instruction(command);
        instruction.lock.lock();
        try {
            //disableSlmControllers();
            controllerClient.instructions.add(instruction);
            instruction.condition.await(5, TimeUnit.SECONDS);
        } catch (InterruptedException ex) {
            showText("Gui: Error: Instruction timed out: " + ex.toString());
        } finally {
            instruction.lock.unlock();
            //enableSlmControllers();
            try {
                if (controllerClient.output.startsWith("Slm: Error: ")) {
                    handleSlmError(controllerClient.output);
                    instructionDone = false;
                } else if (controllerClient.output.startsWith("Arduino: Error: ")) {
                    handleArduinoError(controllerClient.output);
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
     * handles error-massages from the SLM
     *
     * @param error Error-Massage from the SLM
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

    /**
     * handles an error-massage from the Arduino
     *
     * @param error
     */
    private void handleArduinoError(String error) {
        if (error.contains("Gui: Could not find COM port")) {
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
                for (int i = 0; i < controllerClient.slmList.length; i++) {
                    slmComboBox.addItem("[" + i + "]    " + controllerClient.slmList[i]);
                }
            }
            sendSlmInstruction("info");
            if (instructionDone) {
                //slmVersion.setText("SLM-API-Version: " + client.info[0]);
                //slmTime.setText("Timestamp: " + client.info[1]);
                //slmType.setText("Activation type: " + client.info[2]);
                //slmDefault.setText("Default running order: " + slmComboBox.getItemAt(Integer.parseInt(client.info[3])));
                slmSelect.setText("Selected running order: " + slmComboBox.getItemAt(Integer.parseInt(controllerClient.slmInfo[4])));
                //slmRepertoir.setText("Repertoir name: " + client.info[5]);
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
        slmConnectButton = new javax.swing.JButton();
        slmComboBox = new javax.swing.JComboBox<>();
        slmDisconnectButton = new javax.swing.JButton();
        slmRebootButton = new javax.swing.JButton();
        slmSelect = new javax.swing.JLabel();
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
        arduinoLasersLabel = new javax.swing.JLabel();
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
                .addContainerGap()
                .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(slmPanelLayout.createSequentialGroup()
                        .addComponent(slmComboBox, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmSelectButton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmActivateButton, javax.swing.GroupLayout.PREFERRED_SIZE, 85, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(slmDeactivateButton))
                    .addGroup(slmPanelLayout.createSequentialGroup()
                        .addGroup(slmPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(slmPanelLayout.createSequentialGroup()
                                .addComponent(slmConnectButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(slmDisconnectButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(slmRebootButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(slmRefreshButton))
                            .addComponent(slmSelect))
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
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
                        .addComponent(arduinoComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, 81, javax.swing.GroupLayout.PREFERRED_SIZE)
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

        softwarePanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Sync-Controller"));

        syncDelayButton.setText("Set Delay");
        syncDelayButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                syncDelayButtonActionPerformed(evt);
            }
        });

        syncDelayTextField.setText("0");

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
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(arduinoPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(clientServerPanel, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(javax.swing.GroupLayout.Alignment.LEADING, layout.createSequentialGroup()
                        .addComponent(softwarePanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(regPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addComponent(slmPanel, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
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
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(softwarePanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(regPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
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
                            Logger.getLogger(ControllerClientGui.class.getName()).log(Level.SEVERE, null, ex);
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
            int breakTime = Integer.parseInt(arduinoBreakTimeTextField.getText());
            if (breakTime >= 0) {
                startArduinoProgramm((String) arduinoComboBox.getSelectedItem() + breakTime);
            } else {
                throw new NumberFormatException();
            }
        } catch (NumberFormatException ex) {
            startArduinoProgramm((String) arduinoComboBox.getSelectedItem());
        }
    }//GEN-LAST:event_arduinoStartButtonActionPerformed

    private void arduinoStopButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_arduinoStopButtonActionPerformed
        stopArduinoProgramm();
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
        seqDetection.resetChannelBuffers();
        sendArduinoInstruction((String) arduinoComboBox.getSelectedItem() + "xx");
        if (instructionDone) {
            setRGBButtonSelected(false);
        }
    }//GEN-LAST:event_arduinoPhotoButtonActionPerformed

    private void syncDelayButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_syncDelayButtonActionPerformed
        try {
            seqDetection.setSyncDelay(Integer.parseInt(syncDelayTextField.getText()));
        } catch (NumberFormatException e) {
        } catch (DataFormatException ex) {
        }
        syncDelayLabel.setText("Delay: " + seqDetection.getSyncDelay());
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
        seqDetection.resetChannelBuffers();
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
    private javax.swing.JLabel arduinoLasersLabel;
    private javax.swing.JPanel arduinoPanel;
    private javax.swing.JButton arduinoPhotoButton;
    private javax.swing.JToggleButton arduinoRedButton;
    private javax.swing.JButton arduinoStartButton;
    private javax.swing.JButton arduinoStopButton;
    private javax.swing.JPanel clientServerPanel;
    private javax.swing.JButton regCreatorButton;
    private javax.swing.JPanel regPanel;
    public javax.swing.JToggleButton regReconButton;
    public javax.swing.JToggleButton regWfButton;
    private javax.swing.JLabel serverLabel;
    private javax.swing.JButton slmActivateButton;
    private javax.swing.JComboBox<String> slmComboBox;
    private javax.swing.JButton slmConnectButton;
    private javax.swing.JButton slmDeactivateButton;
    private javax.swing.JButton slmDisconnectButton;
    private javax.swing.JPanel slmPanel;
    private javax.swing.JButton slmRebootButton;
    private javax.swing.JButton slmRefreshButton;
    private javax.swing.JLabel slmSelect;
    private javax.swing.JButton slmSelectButton;
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
