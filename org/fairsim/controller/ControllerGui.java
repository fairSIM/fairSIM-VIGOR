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
import java.util.logging.Level;
import java.util.logging.Logger;
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
public class ControllerGui extends javax.swing.JPanel implements ClientGui {

    private final ControllerClient controllerClient;
    private List<CameraClient> camClients;
    private List<Component> slmControllers, arduinoControllers; // cam0Controllers, cam1Controllers, cam2Controllers;
    private static final int CAMCOUNT = 3;
    private static final int ROILENGTH = 4;
    private List<Component>[] camControllers;
    private List<String> arduinoCommands;
    private boolean controllerInstructionDone;
    private boolean[] camInstructionDone;
    String controllerAdress, redCamAdress, greenCamAdress, blueCamAdress, regFolder;
    int TCPPort;
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
    public ControllerGui(Conf.Folder cfg, String[] channelNames, SimSequenceExtractor seqDetection, ReconstructionRunner recRunner) {
        this.seqDetection = seqDetection;
        this.recRunner = recRunner;
        this.channelNames = channelNames;
        try {
            controllerAdress = cfg.getStr("ControllerAdress").val();
        } catch (Conf.EntryNotFoundException ex) {
            controllerAdress = "localhost";
            Tool.error("[fairSIM] No ControllerAdress found. ControllerAdress set to 'localhost'", false);
        }
        try {
            TCPPort = cfg.getInt("TCPPort").val();
        } catch (Conf.EntryNotFoundException ex) {
            TCPPort = 32322;
            Tool.error("[fairSIM] No TCPPort found. TCPPort set to '32322'", false);
        }
        initComponents();
        //ControllerClient.startClient(serverAdress, serverPort, this);
        controllerClient = new ControllerClient(controllerAdress, TCPPort, this);
        controllerClient.start();
        setConnectionLabel(controllerAdress, TCPPort);
        connectCams(cfg);
        initSlm();
        initArduino();
        initCams();
        initSync();
        initReg(cfg);
    }

    /**
     * Initialises the GUI for the SLM
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
     * Initialises the GUI for the arduino
     */
    private void initArduino() {
        arduinoCommands = new ArrayList<>();
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

    private void initCams() {
        camControllers = new List[CAMCOUNT];

        camControllers[0] = new ArrayList<>();
        camControllers[0].add(this.cam0ChannelLabel);
        camControllers[0].add(this.cam0ConfigBox);
        camControllers[0].add(this.cam0ConfigButton);
        camControllers[0].add(this.cam0ExposureButton);
        camControllers[0].add(this.cam0ExposureField);
        camControllers[0].add(this.cam0ExposureLabel);
        camControllers[0].add(this.cam0GroupBox);
        camControllers[0].add(this.cam0MsLabel);
        camControllers[0].add(this.cam0StartButton);
        camControllers[0].add(this.cam0StopButton);
        camControllers[0].add(this.cam0RoiButton);
        camControllers[0].add(this.cam0RoiHField);
        camControllers[0].add(this.cam0RoiLabel);
        camControllers[0].add(this.cam0RoiWField);
        camControllers[0].add(this.cam0RoiXField);
        camControllers[0].add(this.cam0RoiYField);
        disableCamControllers(0);

        camControllers[1] = new ArrayList<>();
        camControllers[1].add(this.cam1ChannelLabel);
        camControllers[1].add(this.cam1ConfigBox);
        camControllers[1].add(this.cam1ConfigButton);
        camControllers[1].add(this.cam1ExposureButton);
        camControllers[1].add(this.cam1ExposureField);
        camControllers[1].add(this.cam1ExposureLabel);
        camControllers[1].add(this.cam1GroupBox);
        camControllers[1].add(this.cam1MsLabel);
        camControllers[1].add(this.cam1StartButton);
        camControllers[1].add(this.cam1StopButton);
        camControllers[1].add(this.cam1RoiButton);
        camControllers[1].add(this.cam1RoiHField);
        camControllers[1].add(this.cam1RoiLabel);
        camControllers[1].add(this.cam1RoiWField);
        camControllers[1].add(this.cam1RoiXField);
        camControllers[1].add(this.cam1RoiYField);
        disableCamControllers(1);

        camControllers[2] = new ArrayList<>();
        camControllers[2].add(this.cam2ChannelLabel);   //0
        camControllers[2].add(this.cam2ConfigBox);      //1
        camControllers[2].add(this.cam2ConfigButton);   //2
        camControllers[2].add(this.cam2ExposureButton); //3
        camControllers[2].add(this.cam2ExposureField);  //4
        camControllers[2].add(this.cam2ExposureLabel);  //5
        camControllers[2].add(this.cam2GroupBox);       //6
        camControllers[2].add(this.cam2MsLabel);        //7
        camControllers[2].add(this.cam2StartButton);    //8
        camControllers[2].add(this.cam2StopButton);     //9
        camControllers[2].add(this.cam2RoiButton);      //10
        camControllers[2].add(this.cam2RoiHField);      //11
        camControllers[2].add(this.cam2RoiLabel);       //12
        camControllers[2].add(this.cam2RoiWField);      //13
        camControllers[2].add(this.cam2RoiXField);      //14
        camControllers[2].add(this.cam2RoiYField);      //15
        disableCamControllers(2);

        camInstructionDone = new boolean[CAMCOUNT];
    }

    private void enableCamButtons(int camId) {
        camControllers[camId].get(2).setEnabled(true);
        camControllers[camId].get(3).setEnabled(true);
        camControllers[camId].get(8).setEnabled(true);
        camControllers[camId].get(9).setEnabled(true);
        camControllers[camId].get(10).setEnabled(true);
    }

    private void disableCamButtons(int camId) {
        camControllers[camId].get(2).setEnabled(false);
        camControllers[camId].get(3).setEnabled(false);
        camControllers[camId].get(8).setEnabled(false);
        camControllers[camId].get(9).setEnabled(false);
        camControllers[camId].get(10).setEnabled(false);
    }

    private void connectCams(Conf.Folder cfg) {
        int len = channelNames.length;
        //readin cam adresses
        String[] adresses = new String[len];
        for (int i = 0; i < len; i++) {
            try {
                Conf.Folder fld = cfg.cd("channel-" + channelNames[i]);
                adresses[i] = fld.getStr("CamAdress").val();
            } catch (Conf.EntryNotFoundException ex) {
                adresses[i] = null;
                Tool.error("[fairSIM] No camera adress found for channel " + channelNames[i], false);
            }
        }
        //check for doublicates
        for (int i = 0; i < len; i++) {
            if (adresses[i] != null) {
                if (adresses[i].equals(controllerAdress)) {
                    adresses[i] = null;
                    Tool.error("[fairSIM] Camera adress of channel '" + channelNames[i] + "' equals controller adress", false);
                } else {
                    for (int j = 0; j < len; j++) {
                        if (i != j && adresses[i].equals(adresses[j])) {
                            adresses[j] = null;
                            Tool.error("[fairSIM] Camera adress of channel '" + channelNames[j] + "' equals camera adress of channel " + channelNames[i], false);
                        }
                    }
                }
            }
        }
        //start camera clients
        camClients = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            if (adresses[i] != null) {
                CameraClient c = new CameraClient(adresses[i], TCPPort, this, channelNames[i]);
                c.start();
                camClients.add(c);
            }
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

    private void enableCamControllers(int camId) {
        for (Component comp : camControllers[camId]) {
            comp.setEnabled(true);
        }
    }

    private void disableCamControllers(int camId) {
        for (Component comp : camControllers[camId]) {
            comp.setEnabled(false);
        }
    }

    /**
     * registers a client at the GUI
     *
     * @param client
     */
    public void registerClient(AbstractClient client) {
        if (client instanceof ControllerClient) {
            //controllerClient = (ControllerClient) client;
            slmConnectButton.setEnabled(true);
            arduinoConnectButton.setEnabled(true);
        } else if (client instanceof CameraClient) {
            for (int i = 0; i < CAMCOUNT; i++) {
                if (client == camClients.get(i)) {
                    javax.swing.JLabel channelLabel = (javax.swing.JLabel) camControllers[i].get(0);
                    channelLabel.setText("Channel: " + camClients.get(i).channelName);
                    updateRoi(i);
                    updateExposure(i);
                    updateGroups(i);
                    enableCamControllers(i);
                    camControllers[i].get(9).setEnabled(false);
                    break;
                }
            }
        }
    }

    private void updateRoi(int camId) {
        sendCamInstruction("get roi", camId);
        int[] rois = camClients.get(camId).rois;
        javax.swing.JLabel roiLabel = (javax.swing.JLabel) camControllers[camId].get(12);
        roiLabel.setText("ROI: " + rois[0] + ", " + rois[1] + ", " + rois[2] + ", " + rois[3]);
    }

    private void updateExposure(int camId) {
        sendCamInstruction("get exposure", camId);
        javax.swing.JLabel exposureLabel = (javax.swing.JLabel) camControllers[camId].get(5);
        exposureLabel.setText("Exposure Time: " + camClients.get(camId).exposure);
    }

    private void updateGroups(int camId) {
        sendCamInstruction("get groups", camId);
        String[] groups = camClients.get(camId).getGroupArray();
        javax.swing.JComboBox<String> groupBox = (javax.swing.JComboBox<String>) camControllers[camId].get(6);
        groupBox.removeAllItems();
        for (String s : groups) {
            groupBox.addItem(s);
        }
        updateConfigs(camId, groupBox.getSelectedIndex());
    }

    private void updateConfigs(int camId, int groupId) {
        String[] configs = camClients.get(camId).getConfigArray(groupId);
        javax.swing.JComboBox<String> configBox = (javax.swing.JComboBox<String>) camControllers[camId].get(1);
        configBox.removeAllItems();
        for (String s : configs) {
            configBox.addItem(s);
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
            slmSelect.setText("Selected running order: -");
            slmComboBox.removeAllItems();
            slmConnectButton.setEnabled(false);
            disableArduinoControllers();
        } else if (client instanceof CameraClient) {
            for (int i = 0; i < CAMCOUNT; i++) {
                if (client == camClients.get(i)) {
                    disableCamControllers(i);
                }
                break;
            }
        }
    }

    /**
     * Sends a command to the SLM
     *
     * @param command
     */
    void sendSlmInstruction(String command) {
        sendControllerInstruction("slm->" + command);
    }

    /**
     * Sends a command to the arduino
     *
     * @param command
     */
    void sendArduinoInstruction(String command) {
        sendControllerInstruction("arduino->" + command);
    }

    /**
     * Sends a command for the server to the client
     *
     * @param command command for the server
     */
    void sendControllerInstruction(String command) {
        controllerInstructionDone = true;
        controllerClient.addInstruction(command);
    }

    void sendCamInstruction(String command, int camId) {
        camInstructionDone[camId] = true;
        camClients.get(camId).addInstruction(command);
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
    void handleArduinoError(String error) {
        controllerInstructionDone = false;
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

    void handleCamError(String error, CameraClient client) {
        for (int i = 0; i < CAMCOUNT; i++) {
            if (client == camClients.get(i)) {
                camInstructionDone[i] = false;
                //disableCamControllers(i);
                showText("Gui: Error with camera channel '" + client.channelName + "' occurred");
                showText(error);
            }
        }
    }

    /**
     * refreshes the informations shown in the GUI
     */
    void refreshSlmInfo() {
        try {
            sendSlmInstruction("rolist");
            if (controllerInstructionDone) {
                slmComboBox.removeAllItems();
                for (int i = 0; i < controllerClient.slmList.length; i++) {
                    slmComboBox.addItem("[" + i + "]    " + controllerClient.slmList[i]);
                }
            }
            sendSlmInstruction("info");
            if (controllerInstructionDone) {
                //slmVersion.setText("SLM-API-Version: " + client.info[0]);
                //slmTime.setText("Timestamp: " + client.info[1]);
                //slmType.setText("Activation type: " + client.info[2]);
                //slmDefault.setText("Default running order: " + slmComboBox.getItemAt(Integer.parseInt(client.info[3])));
                slmSelect.setText("Selected running order: " + slmComboBox.getItemAt(Integer.parseInt(controllerClient.slmInfo[4])));
                //slmRepertoir.setText("Repertoir name: " + client.info[5]);
            }
        } catch (NullPointerException ex) {
            System.err.println("Error while refreshing SLM-GUI");
            showText("[fairSIM] Error while refreshing SLM-GUI");
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
        camPanel = new javax.swing.JPanel();
        cam0ChannelLabel = new javax.swing.JLabel();
        cam0RoiButton = new javax.swing.JButton();
        cam0RoiXField = new javax.swing.JTextField();
        cam0RoiYField = new javax.swing.JTextField();
        cam0RoiWField = new javax.swing.JTextField();
        cam0RoiHField = new javax.swing.JTextField();
        cam0ExposureButton = new javax.swing.JButton();
        cam0ExposureField = new javax.swing.JTextField();
        cam0MsLabel = new javax.swing.JLabel();
        cam0GroupBox = new javax.swing.JComboBox<>();
        cam0ConfigBox = new javax.swing.JComboBox<>();
        cam0ConfigButton = new javax.swing.JButton();
        cam0RoiLabel = new javax.swing.JLabel();
        cam0ExposureLabel = new javax.swing.JLabel();
        cam1ChannelLabel = new javax.swing.JLabel();
        cam1RoiLabel = new javax.swing.JLabel();
        cam1ExposureLabel = new javax.swing.JLabel();
        cam1RoiXField = new javax.swing.JTextField();
        cam1RoiYField = new javax.swing.JTextField();
        cam1RoiWField = new javax.swing.JTextField();
        cam1RoiHField = new javax.swing.JTextField();
        cam1RoiButton = new javax.swing.JButton();
        cam1ExposureField = new javax.swing.JTextField();
        cam1MsLabel = new javax.swing.JLabel();
        cam1ExposureButton = new javax.swing.JButton();
        cam1GroupBox = new javax.swing.JComboBox<>();
        cam1ConfigBox = new javax.swing.JComboBox<>();
        cam1ConfigButton = new javax.swing.JButton();
        cam2ChannelLabel = new javax.swing.JLabel();
        cam2RoiXField = new javax.swing.JTextField();
        cam2RoiYField = new javax.swing.JTextField();
        cam2RoiWField = new javax.swing.JTextField();
        cam2RoiHField = new javax.swing.JTextField();
        cam2RoiLabel = new javax.swing.JLabel();
        cam2RoiButton = new javax.swing.JButton();
        cam2ExposureField = new javax.swing.JTextField();
        cam2ExposureLabel = new javax.swing.JLabel();
        cam2MsLabel = new javax.swing.JLabel();
        cam2ExposureButton = new javax.swing.JButton();
        cam2GroupBox = new javax.swing.JComboBox<>();
        cam2ConfigBox = new javax.swing.JComboBox<>();
        cam2ConfigButton = new javax.swing.JButton();
        cam0StopButton = new javax.swing.JButton();
        cam0StartButton = new javax.swing.JButton();
        cam1StopButton = new javax.swing.JButton();
        cam1StartButton = new javax.swing.JButton();
        cam2StopButton = new javax.swing.JButton();
        cam2StartButton = new javax.swing.JButton();

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

        camPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Camera-Controller"));

        cam0ChannelLabel.setText("Channel: - ");

        cam0RoiButton.setText("Set ROI");
        cam0RoiButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam0RoiButtonActionPerformed(evt);
            }
        });

        cam0RoiXField.setText("751");

        cam0RoiYField.setText("765");

        cam0RoiWField.setText("520");

        cam0RoiHField.setText("520");

        cam0ExposureButton.setText("Set Exposure");
        cam0ExposureButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam0ExposureButtonActionPerformed(evt);
            }
        });

        cam0ExposureField.setText("2500");

        cam0MsLabel.setText("ms");

        cam0GroupBox.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                cam0GroupBoxItemStateChanged(evt);
            }
        });

        cam0ConfigButton.setText("Set Config");
        cam0ConfigButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam0ConfigButtonActionPerformed(evt);
            }
        });

        cam0RoiLabel.setText("ROI: -");

        cam0ExposureLabel.setText("Exposure Time: - ");

        cam1ChannelLabel.setText("Channel: - ");

        cam1RoiLabel.setText("ROI: -");

        cam1ExposureLabel.setText("Exposure Time: - ");

        cam1RoiXField.setText("751");

        cam1RoiYField.setText("765");

        cam1RoiWField.setText("520");

        cam1RoiHField.setText("520");

        cam1RoiButton.setText("Set ROI");
        cam1RoiButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam1RoiButtonActionPerformed(evt);
            }
        });

        cam1ExposureField.setText("2500");

        cam1MsLabel.setText("ms");

        cam1ExposureButton.setText("Set Exposure");
        cam1ExposureButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam1ExposureButtonActionPerformed(evt);
            }
        });

        cam1GroupBox.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                cam1GroupBoxItemStateChanged(evt);
            }
        });

        cam1ConfigButton.setText("Set Config");
        cam1ConfigButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam1ConfigButtonActionPerformed(evt);
            }
        });

        cam2ChannelLabel.setText("Channel: - ");

        cam2RoiXField.setText("751");

        cam2RoiYField.setText("765");

        cam2RoiWField.setText("520");

        cam2RoiHField.setText("520");

        cam2RoiLabel.setText("ROI: -");

        cam2RoiButton.setText("Set ROI");
        cam2RoiButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam2RoiButtonActionPerformed(evt);
            }
        });

        cam2ExposureField.setText("2500");

        cam2ExposureLabel.setText("Exposure Time: - ");

        cam2MsLabel.setText("ms");

        cam2ExposureButton.setText("Set Exposure");
        cam2ExposureButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam2ExposureButtonActionPerformed(evt);
            }
        });

        cam2GroupBox.addItemListener(new java.awt.event.ItemListener() {
            public void itemStateChanged(java.awt.event.ItemEvent evt) {
                cam2GroupBoxItemStateChanged(evt);
            }
        });

        cam2ConfigButton.setText("Set Config");
        cam2ConfigButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam2ConfigButtonActionPerformed(evt);
            }
        });

        cam0StopButton.setText("Stop Acquisition");
        cam0StopButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam0StopButtonActionPerformed(evt);
            }
        });

        cam0StartButton.setText("Start Acquisition");
        cam0StartButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam0StartButtonActionPerformed(evt);
            }
        });

        cam1StopButton.setText("Stop Acquisition");
        cam1StopButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam1StopButtonActionPerformed(evt);
            }
        });

        cam1StartButton.setText("Start Acquisition");
        cam1StartButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam1StartButtonActionPerformed(evt);
            }
        });

        cam2StopButton.setText("Stop Acquisition");
        cam2StopButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam2StopButtonActionPerformed(evt);
            }
        });

        cam2StartButton.setText("Start Acquisition");
        cam2StartButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cam2StartButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout camPanelLayout = new javax.swing.GroupLayout(camPanel);
        camPanel.setLayout(camPanelLayout);
        camPanelLayout.setHorizontalGroup(
            camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(camPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(camPanelLayout.createSequentialGroup()
                        .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam2RoiXField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam2RoiYField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam2RoiWField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(cam2ChannelLabel))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam2RoiHField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam2RoiButton)
                                .addGap(18, 18, 18)
                                .addComponent(cam2ExposureField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam2MsLabel))
                            .addComponent(cam2RoiLabel))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam2ExposureLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(cam2StartButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam2StopButton))
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam2ExposureButton)
                                .addGap(18, 18, 18)
                                .addComponent(cam2GroupBox, javax.swing.GroupLayout.PREFERRED_SIZE, 75, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam2ConfigBox, javax.swing.GroupLayout.PREFERRED_SIZE, 75, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam2ConfigButton))))
                    .addGroup(camPanelLayout.createSequentialGroup()
                        .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam1RoiXField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam1RoiYField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam1RoiWField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(cam1ChannelLabel))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam1RoiHField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam1RoiButton)
                                .addGap(18, 18, 18)
                                .addComponent(cam1ExposureField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam1MsLabel))
                            .addComponent(cam1RoiLabel))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam1ExposureLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(cam1StartButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam1StopButton))
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam1ExposureButton)
                                .addGap(18, 18, 18)
                                .addComponent(cam1GroupBox, javax.swing.GroupLayout.PREFERRED_SIZE, 75, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam1ConfigBox, javax.swing.GroupLayout.PREFERRED_SIZE, 75, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam1ConfigButton))))
                    .addGroup(camPanelLayout.createSequentialGroup()
                        .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam0RoiXField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0RoiYField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0RoiWField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(cam0ChannelLabel))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam0RoiHField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0RoiButton)
                                .addGap(18, 18, 18)
                                .addComponent(cam0ExposureField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0MsLabel))
                            .addComponent(cam0RoiLabel))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam0ExposureLabel)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(cam0StartButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0StopButton))
                            .addGroup(camPanelLayout.createSequentialGroup()
                                .addComponent(cam0ExposureButton)
                                .addGap(18, 18, 18)
                                .addComponent(cam0GroupBox, javax.swing.GroupLayout.PREFERRED_SIZE, 75, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0ConfigBox, javax.swing.GroupLayout.PREFERRED_SIZE, 75, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(cam0ConfigButton)))))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        camPanelLayout.setVerticalGroup(
            camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(camPanelLayout.createSequentialGroup()
                .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cam0ChannelLabel)
                    .addComponent(cam0RoiLabel)
                    .addComponent(cam0ExposureLabel)
                    .addComponent(cam0StopButton)
                    .addComponent(cam0StartButton))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cam0RoiButton)
                    .addComponent(cam0RoiXField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0RoiYField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0RoiWField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0RoiHField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0ExposureButton)
                    .addComponent(cam0ExposureField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0MsLabel)
                    .addComponent(cam0GroupBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0ConfigBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam0ConfigButton))
                .addGap(18, 18, 18)
                .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cam1ChannelLabel)
                    .addComponent(cam1RoiLabel)
                    .addComponent(cam1ExposureLabel)
                    .addComponent(cam1StopButton)
                    .addComponent(cam1StartButton))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cam1RoiButton)
                    .addComponent(cam1RoiXField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam1RoiYField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam1RoiWField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam1RoiHField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam1ExposureButton)
                    .addComponent(cam1ExposureField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam1MsLabel)
                    .addComponent(cam1GroupBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam1ConfigBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam1ConfigButton))
                .addGap(18, 18, 18)
                .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cam2ChannelLabel)
                    .addComponent(cam2RoiLabel)
                    .addComponent(cam2ExposureLabel)
                    .addComponent(cam2StopButton)
                    .addComponent(cam2StartButton))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(camPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cam2RoiButton)
                    .addComponent(cam2RoiXField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam2RoiYField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam2RoiWField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam2RoiHField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam2ExposureButton)
                    .addComponent(cam2ExposureField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam2MsLabel)
                    .addComponent(cam2GroupBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam2ConfigBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(cam2ConfigButton))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(camPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(arduinoPanel, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(clientServerPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(softwarePanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(regPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addComponent(slmPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(slmPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(arduinoPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(camPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(19, 19, 19)
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
        if (controllerInstructionDone) {
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
                            Logger.getLogger(ControllerGui.class.getName()).log(Level.SEVERE, null, ex);
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
        if (controllerInstructionDone) {
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

    private void cam0StartButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam0StartButtonActionPerformed
        startCam(0);
    }//GEN-LAST:event_cam0StartButtonActionPerformed

    private void cam0StopButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam0StopButtonActionPerformed
        stopCam(0);
    }//GEN-LAST:event_cam0StopButtonActionPerformed

    private void cam0RoiButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam0RoiButtonActionPerformed
        setRoi(0);
    }//GEN-LAST:event_cam0RoiButtonActionPerformed

    private void cam0ExposureButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam0ExposureButtonActionPerformed
        setExposureTime(0);
    }//GEN-LAST:event_cam0ExposureButtonActionPerformed

    private void cam0ConfigButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam0ConfigButtonActionPerformed
        setConfig(0);
    }//GEN-LAST:event_cam0ConfigButtonActionPerformed

    private void cam0GroupBoxItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_cam0GroupBoxItemStateChanged
        groupBoxSelected(0);
    }//GEN-LAST:event_cam0GroupBoxItemStateChanged

    private void cam1StartButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam1StartButtonActionPerformed
        startCam(1);
    }//GEN-LAST:event_cam1StartButtonActionPerformed

    private void cam1StopButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam1StopButtonActionPerformed
        stopCam(1);
    }//GEN-LAST:event_cam1StopButtonActionPerformed

    private void cam1RoiButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam1RoiButtonActionPerformed
        setRoi(1);
    }//GEN-LAST:event_cam1RoiButtonActionPerformed

    private void cam1ExposureButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam1ExposureButtonActionPerformed
        setExposureTime(1);
    }//GEN-LAST:event_cam1ExposureButtonActionPerformed

    private void cam1ConfigButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam1ConfigButtonActionPerformed
        setConfig(1);
    }//GEN-LAST:event_cam1ConfigButtonActionPerformed

    private void cam2StartButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam2StartButtonActionPerformed
        startCam(2);
    }//GEN-LAST:event_cam2StartButtonActionPerformed

    private void cam2StopButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam2StopButtonActionPerformed
        stopCam(2);
    }//GEN-LAST:event_cam2StopButtonActionPerformed

    private void cam2RoiButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam2RoiButtonActionPerformed
        setRoi(2);
    }//GEN-LAST:event_cam2RoiButtonActionPerformed

    private void cam2ExposureButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam2ExposureButtonActionPerformed
        setExposureTime(2);
    }//GEN-LAST:event_cam2ExposureButtonActionPerformed

    private void cam2ConfigButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cam2ConfigButtonActionPerformed
        setConfig(2);
    }//GEN-LAST:event_cam2ConfigButtonActionPerformed

    private void cam1GroupBoxItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_cam1GroupBoxItemStateChanged
        groupBoxSelected(1);
    }//GEN-LAST:event_cam1GroupBoxItemStateChanged

    private void cam2GroupBoxItemStateChanged(java.awt.event.ItemEvent evt) {//GEN-FIRST:event_cam2GroupBoxItemStateChanged
        groupBoxSelected(2);
    }//GEN-LAST:event_cam2GroupBoxItemStateChanged

    private void startCam(int camId) {
        sendCamInstruction("start", camId);
        if (camInstructionDone[camId]) {
            disableCamButtons(camId);
            switch (camId) {
                case 0:
                    cam0StopButton.setEnabled(true);
                case 1:
                    cam1StopButton.setEnabled(true);
                case 2:
                    cam2StopButton.setEnabled(true);
            }
        }
    }

    private void stopCam(int camId) {
        sendCamInstruction("stop", camId);
        if (camInstructionDone[camId]) {
            enableCamButtons(camId);
            switch (camId) {
                case 0:
                    cam0StopButton.setEnabled(false);
                case 1:
                    cam1StopButton.setEnabled(false);
                case 2:
                    cam2StopButton.setEnabled(false);
            }
        }
    }

    private void setRoi(int camId) {
        try {
            int[] roi = getRoi(camId);
            String sRoi = Tool.encodeArray("set roi", roi);
            sendCamInstruction(sRoi, camId);
            if (camInstructionDone[camId]) {
                updateRoi(camId);
            }
        } catch (NumberFormatException ex) {
        }
    }

    private void setExposureTime(int camId) {
        try {
            javax.swing.JTextField exposureField = (javax.swing.JTextField) camControllers[camId].get(4);
            String exposureString = exposureField.getText();
            double exposureTime = Double.parseDouble(exposureString);
            sendCamInstruction("set exposure;" + exposureTime, camId);
            if (camInstructionDone[camId]) {
                updateExposure(camId);
            }
        } catch (NumberFormatException ex) {
        }
    }

    private void setConfig(int camId) {
        javax.swing.JComboBox<String> groupBox = (javax.swing.JComboBox<String>) camControllers[camId].get(6);
        javax.swing.JComboBox<String> configBox = (javax.swing.JComboBox<String>) camControllers[camId].get(1);
        int[] ids = new int[2];
        ids[0] = groupBox.getSelectedIndex();
        ids[1] = configBox.getSelectedIndex();
        String stringIds = Tool.encodeArray("set config", ids);
        sendCamInstruction(stringIds, camId);
    }

    private void groupBoxSelected(int camId) {
        javax.swing.JComboBox<String> groupBox = (javax.swing.JComboBox<String>) camControllers[camId].get(6);
        int groupId = groupBox.getSelectedIndex();
        if (groupId >= 0) {
            this.updateConfigs(camId, groupId);
        }
    }

    private int[] getRoi(int camId) throws NumberFormatException {
        javax.swing.JTextField x = (javax.swing.JTextField) camControllers[camId].get(14);
        javax.swing.JTextField y = (javax.swing.JTextField) camControllers[camId].get(15);
        javax.swing.JTextField w = (javax.swing.JTextField) camControllers[camId].get(13);
        javax.swing.JTextField h = (javax.swing.JTextField) camControllers[camId].get(11);
        int[] roi = new int[ROILENGTH];
        roi[0] = Integer.parseInt(x.getText());
        roi[1] = Integer.parseInt(y.getText());
        roi[2] = Integer.parseInt(w.getText());
        roi[3] = Integer.parseInt(h.getText());
        return roi;
    }

    private void setRGBButtonSelected(boolean b) {
        arduinoRedButton.setSelected(b);
        arduinoGreenButton.setSelected(b);
        arduinoBlueButton.setSelected(b);
    }

    private void stopArduinoProgramm() {
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

    private void startArduinoProgramm(String command) {
        seqDetection.resetChannelBuffers();
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
     * Connects server and SLM
     */
    private void connectSLM() {
        sendSlmInstruction("connect");
        if (controllerInstructionDone) {
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
        if (controllerInstructionDone) {
            disableSlmControllers();
            slmConnectButton.setEnabled(true);
        }
    }

    private void connectArduino() {
        sendArduinoInstruction("connect");
        if (controllerInstructionDone) {
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
        if (controllerInstructionDone) {
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
    private javax.swing.JLabel cam0ChannelLabel;
    private javax.swing.JComboBox<String> cam0ConfigBox;
    private javax.swing.JButton cam0ConfigButton;
    private javax.swing.JButton cam0ExposureButton;
    private javax.swing.JTextField cam0ExposureField;
    private javax.swing.JLabel cam0ExposureLabel;
    private javax.swing.JComboBox<String> cam0GroupBox;
    private javax.swing.JLabel cam0MsLabel;
    private javax.swing.JButton cam0RoiButton;
    private javax.swing.JTextField cam0RoiHField;
    private javax.swing.JLabel cam0RoiLabel;
    private javax.swing.JTextField cam0RoiWField;
    private javax.swing.JTextField cam0RoiXField;
    private javax.swing.JTextField cam0RoiYField;
    private javax.swing.JButton cam0StartButton;
    private javax.swing.JButton cam0StopButton;
    private javax.swing.JLabel cam1ChannelLabel;
    private javax.swing.JComboBox<String> cam1ConfigBox;
    private javax.swing.JButton cam1ConfigButton;
    private javax.swing.JButton cam1ExposureButton;
    private javax.swing.JTextField cam1ExposureField;
    private javax.swing.JLabel cam1ExposureLabel;
    private javax.swing.JComboBox<String> cam1GroupBox;
    private javax.swing.JLabel cam1MsLabel;
    private javax.swing.JButton cam1RoiButton;
    private javax.swing.JTextField cam1RoiHField;
    private javax.swing.JLabel cam1RoiLabel;
    private javax.swing.JTextField cam1RoiWField;
    private javax.swing.JTextField cam1RoiXField;
    private javax.swing.JTextField cam1RoiYField;
    private javax.swing.JButton cam1StartButton;
    private javax.swing.JButton cam1StopButton;
    private javax.swing.JLabel cam2ChannelLabel;
    private javax.swing.JComboBox<String> cam2ConfigBox;
    private javax.swing.JButton cam2ConfigButton;
    private javax.swing.JButton cam2ExposureButton;
    private javax.swing.JTextField cam2ExposureField;
    private javax.swing.JLabel cam2ExposureLabel;
    private javax.swing.JComboBox<String> cam2GroupBox;
    private javax.swing.JLabel cam2MsLabel;
    private javax.swing.JButton cam2RoiButton;
    private javax.swing.JTextField cam2RoiHField;
    private javax.swing.JLabel cam2RoiLabel;
    private javax.swing.JTextField cam2RoiWField;
    private javax.swing.JTextField cam2RoiXField;
    private javax.swing.JTextField cam2RoiYField;
    private javax.swing.JButton cam2StartButton;
    private javax.swing.JButton cam2StopButton;
    private javax.swing.JPanel camPanel;
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
