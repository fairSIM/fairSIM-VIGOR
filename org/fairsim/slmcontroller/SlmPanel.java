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

package org.fairsim.slmcontroller;

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
public class SlmPanel extends javax.swing.JPanel {

    Client slmClient;
    private List<Component> controllers;
    private boolean instructionDone;
    String serverAdress;
    int serverPort;

    /**
     * Creates new form SlmPanel
     * @param cfg Configuration settings
     */
    public SlmPanel(Conf.Folder cfg) {
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
        
        Client.startClient(serverAdress, serverPort , this);

        controllers = new ArrayList<>();

        controllers.add(roComboBox);
        controllers.add(deactivateButton);
        controllers.add(activateButton);
        controllers.add(selectButton);
        controllers.add(disconnectButton);
        controllers.add(refreshButton);
        controllers.add(rebootButton);

        disableControllers();
        connectButton.setEnabled(false);
    }

    /**
     * Shows a new text-line in the text-field
     * @param text String that shoulds be displayed in the text-field
     */
    void showText(String text) {
        textArea1.append(text + "\n");
    }
    
    /**
     * Sets lable on top of the gui
     * @param adress server adress
     * @param port server port
     */
    void setConnectionLabel(String adress, int port) {
        serverLabel.setText("Server: " + adress + ":" + port);
    }

    /**
     * enables buttons of the GUI without connect-button
     */
    private void enableControllers() {
        for (Component comp : controllers) {
            comp.setEnabled(true);
        }
    }

    /**
     * disables buttons of the GUI without connect-button
     */
    void disableControllers() {
        for (Component comp : controllers) {
            comp.setEnabled(false);
        }
    }

    void registerClient(Client slmClient) {
        this.slmClient = slmClient;
        showText("SlmClient successfully registerd at the SlmGui");
        connectButton.setEnabled(true);
    }

    void unregisterClient() {
        showText("SlmClient unregisterd at the SlmGui");
        disableControllers();
        jLabel0.setText("SLM-API-Version -: ");
        jLabel1.setText("Timestamp: -: ");
        jLabel2.setText("Activation type: -");
        jLabel3.setText("Default running order: -");
        jLabel4.setText("Selected running order: -");
        jLabel5.setText("Repertoir name: -");
        roComboBox.removeAllItems();
        connectButton.setEnabled(false);
    }

    /**
     * Sends a command for the server to the client
     * @param command command for the server
     */
    void sendInstruction(String command) {
        Instruction slmInstruction = new Instruction(command);
        slmInstruction.lock.lock();
        try {
            disableControllers();
            slmClient.instructions.add(slmInstruction);
            slmInstruction.condition.await(2, TimeUnit.SECONDS);
        } catch (InterruptedException ex) {
            showText("GUI Error: Instruction timed out");
        } finally {
            slmInstruction.lock.unlock();
            enableControllers();
            try {
                if (slmClient.output.startsWith("Error: ")) {
                    handleError(slmClient.output);
                    instructionDone = false;
                } else {
                    instructionDone = true;
                }
            } catch (NullPointerException ex) {
                showText("Something went wrong");
                disableControllers();
                connectButton.setEnabled(false);
            }
        }
    }

    /**
     * handles error-massages from the server
     * @param error Error-Massage from the server
     */
    private void handleError(String error) {
        //showText(error);
        int code = Integer.parseInt(error.split("  ;  ")[1].split(": ")[1]);
        if (code == 12) {
            disconnectSLM();
            showText("Connection to the SLM lost");
        } else if (code == 7) {
            disableControllers();
            connectButton.setEnabled(true);
            showText("No connection to the SLM");
        } else if (code == 8) {
            disconnectSLM();
            connectSLM();
        }
    }

    /**
     * refreshes the informations shown in the GUI
     */
    void refresh() {
        try {
            sendInstruction("rolist");
            if (instructionDone) {
                for (int i = 0; i < slmClient.roList.length; i++) {
                    roComboBox.addItem("[" + i + "]    " + slmClient.roList[i]);
                }
            }
            sendInstruction("info");
            if (instructionDone) {
                jLabel0.setText("SLM-API-Version: " + slmClient.info[0]);
                jLabel1.setText(slmClient.info[1]);
                jLabel2.setText("Activation type: " + slmClient.info[2]);
                jLabel3.setText("Default running order: " + roComboBox.getItemAt(Integer.parseInt(slmClient.info[3])));
                jLabel4.setText("Selected running order: " + roComboBox.getItemAt(Integer.parseInt(slmClient.info[4])));
                jLabel5.setText("Repertoir name: " + slmClient.info[5]);
            }
        } catch (NullPointerException ex) {
            System.err.println("Error while refreshing SLM-GUI");
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

        jLabel3 = new javax.swing.JLabel();
        jLabel4 = new javax.swing.JLabel();
        jLabel5 = new javax.swing.JLabel();
        rebootButton = new javax.swing.JButton();
        connectButton = new javax.swing.JButton();
        disconnectButton = new javax.swing.JButton();
        roComboBox = new javax.swing.JComboBox<>();
        textArea1 = new java.awt.TextArea();
        activateButton = new javax.swing.JButton();
        deactivateButton = new javax.swing.JButton();
        selectButton = new javax.swing.JButton();
        refreshButton = new javax.swing.JButton();
        jLabel0 = new javax.swing.JLabel();
        jLabel1 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        serverLabel = new javax.swing.JLabel();

        jLabel3.setText("Default running order: -");

        jLabel4.setText("Selected running order: -");

        jLabel5.setText("Repertoir name: -");

        rebootButton.setText("(Reboot SLM)");
        rebootButton.setToolTipText("Maybe it works, maybe not");
        rebootButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                rebootButtonActionPerformed(evt);
            }
        });

        connectButton.setText("Connect SLM");
        connectButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                connectButtonActionPerformed(evt);
            }
        });

        disconnectButton.setText("Disconnect SLM");
        disconnectButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                disconnectButtonActionPerformed(evt);
            }
        });

        roComboBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                roComboBoxActionPerformed(evt);
            }
        });

        activateButton.setText("Aktivate");
        activateButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                activateButtonActionPerformed(evt);
            }
        });

        deactivateButton.setText("Deactivate");
        deactivateButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                deactivateButtonActionPerformed(evt);
            }
        });

        selectButton.setText("Set Selected");
        selectButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                selectButtonActionPerformed(evt);
            }
        });

        refreshButton.setText("Refresh GUI");
        refreshButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                refreshButtonActionPerformed(evt);
            }
        });

        jLabel0.setText("SLM-API-Version: -");

        jLabel1.setText("Timestamp: -");

        jLabel2.setText("Activation type: -");

        serverLabel.setFont(new java.awt.Font("Tahoma", 1, 14)); // NOI18N
        serverLabel.setText("Server: -");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                    .addComponent(activateButton, javax.swing.GroupLayout.PREFERRED_SIZE, 85, javax.swing.GroupLayout.PREFERRED_SIZE)
                                    .addComponent(deactivateButton, javax.swing.GroupLayout.Alignment.LEADING))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(layout.createSequentialGroup()
                                        .addComponent(roComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, 290, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                        .addComponent(selectButton))
                                    .addComponent(jLabel3)
                                    .addComponent(jLabel4)
                                    .addComponent(jLabel2)))
                            .addComponent(textArea1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(jLabel5)
                            .addComponent(jLabel1)
                            .addComponent(jLabel0))
                        .addContainerGap())
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(connectButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(disconnectButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(rebootButton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(refreshButton))
                            .addComponent(serverLabel))
                        .addGap(0, 0, Short.MAX_VALUE))))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(serverLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(3, 3, 3)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(connectButton)
                    .addComponent(disconnectButton)
                    .addComponent(rebootButton)
                    .addComponent(refreshButton))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(activateButton)
                        .addGap(3, 3, 3)
                        .addComponent(deactivateButton))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(12, 12, 12)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(roComboBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(selectButton))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(jLabel3)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jLabel4)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabel2)
                .addGap(18, 18, 18)
                .addComponent(jLabel0)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabel1)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabel5)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(textArea1, javax.swing.GroupLayout.DEFAULT_SIZE, 224, Short.MAX_VALUE)
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    private void connectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_connectButtonActionPerformed
        connectSLM();

    }//GEN-LAST:event_connectButtonActionPerformed

    /**
     * Connects server and SLM
     */
    private void connectSLM() {
        sendInstruction("connect");
        if (instructionDone) {
            refresh();
            enableControllers();
            connectButton.setEnabled(false);
        }
    }

    /**
     * Disconnects server and SLM
     */
    private void disconnectSLM() {
        sendInstruction("disconnect");
        if (instructionDone) {
            disableControllers();
            connectButton.setEnabled(true);
        }
    }

    private void disconnectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_disconnectButtonActionPerformed
        disconnectSLM();
    }//GEN-LAST:event_disconnectButtonActionPerformed

    private void roComboBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_roComboBoxActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_roComboBoxActionPerformed

    private void activateButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_activateButtonActionPerformed
        sendInstruction("activate");
    }//GEN-LAST:event_activateButtonActionPerformed

    private void deactivateButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_deactivateButtonActionPerformed
        sendInstruction("deactivate");
    }//GEN-LAST:event_deactivateButtonActionPerformed

    private void selectButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_selectButtonActionPerformed
        sendInstruction(Integer.toString(roComboBox.getSelectedIndex()));
        refresh();
    }//GEN-LAST:event_selectButtonActionPerformed

    private void refreshButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_refreshButtonActionPerformed
        refresh();
    }//GEN-LAST:event_refreshButtonActionPerformed

    private void rebootButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_rebootButtonActionPerformed
        sendInstruction("reboot");
        if (instructionDone) {
            disableControllers();
            Thread timer = new Thread(new Runnable() {
                @Override
                public void run() {
                    showText("Wait 20 Seconds until reboot finnished");
                    for (int time = 20; time > 0; time--) {
                        textArea1.append(time + "... ");
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException ex) {
                            Logger.getLogger(SlmPanel.class.getName()).log(Level.SEVERE, null, ex);
                        }
                    }
                    textArea1.append("\n");
                    enableControllers();
                }
            });
            timer.start();
        }

    }//GEN-LAST:event_rebootButtonActionPerformed


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton activateButton;
    private javax.swing.JButton connectButton;
    private javax.swing.JButton deactivateButton;
    private javax.swing.JButton disconnectButton;
    private javax.swing.JLabel jLabel0;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JButton rebootButton;
    private javax.swing.JButton refreshButton;
    private javax.swing.JComboBox<String> roComboBox;
    private javax.swing.JButton selectButton;
    private javax.swing.JLabel serverLabel;
    private java.awt.TextArea textArea1;
    // End of variables declaration//GEN-END:variables
}
