/*//GEN-LINE:variables
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
package org.fairsim.cameraplugin;

import java.awt.Color;
import javax.swing.JFrame;
import org.fairsim.controller.AbstractServer;
import org.fairsim.sim_gui.PlainImageDisplay;

/**
 *
 * @author m.lachetta
 */
public class CameraServerGui extends javax.swing.JFrame implements AbstractServer.ServerGui {
    private JFrame viewFrame;
    PlainImageDisplay view;
    CameraServer cs;
    private int viewWidth, viewHeight;
    private boolean refreshing;
    private final CameraController cc;
    private final Color defaultColor;
    
    /**
     * Creates new form CameraGui
     */
    public CameraServerGui(int width, int height, CameraController cc) {
        initComponents();
        refreshing = false;
        this.cc = cc;
        initView(width, height);
        cs = CameraServer.startCameraServer(this, cc);
        int[] channels = cc.getChannels();
        for (int ch : channels) {
            setTitle(getTitle() + " - " + ch);
        }
        startButton.setEnabled(true);
        stopButton.setEnabled(false);
        setVisible(true);
        defaultColor = queuingPanel.getBackground();
    }
    
    void setQueuingColor(Color c) {
        queuingPanel.setBackground(c);
    }
    
    void resetQueuingColor() {
        queuingPanel.setBackground(defaultColor);
    }
    
    void setSendingColor(Color c) {
        sendingPanel.setBackground(c);
    }
    
    void resetSendingColor() {
        sendingPanel.setBackground(defaultColor);
    }
    
    void setFps(double fps) {
        fpsLabel.setText("FPS: " + Math.round(fps * 1000) / 1000.0);
    }
    
    void resetFps() {
        fpsLabel.setText("FPS: -");
    }
    
    /**
     * disposes and creates a new view frame, called after changing cameras ROI
     * @param width new image width
     * @param height new image height
     */
    void refreshView(int width, int height) {
        if (viewWidth != width || viewHeight != height) {
            refreshing = true;
            viewFrame.dispose();
            viewFrame = null;
            view = null;
            initView(width, height);
            viewFrame.toFront();
            refreshing = false;
        }
    }
    
    /**
     * closes the whole micro manager pluging including all threads
     */
    public void closeWholePlugin() {
        cc.close();
        cs.close();
        dispose();
        viewFrame.dispose();
    }
    
    /**
     * initializes the view frame
     * @param width image width
     * @param height image height
     */
    private void initView(int width, int height) {
        viewWidth = width;
        viewHeight = height;
        viewFrame = new JFrame("View");
        String[] channels = new String[cc.getChannels().length];
        for (int i = 0; i < channels.length; i++) {
            channels[i] = String.valueOf(cc.getChannels()[i]);
        }
        view = new PlainImageDisplay(channels.length , viewWidth, viewHeight, channels);
        viewFrame.add(view.getPanel());
        viewFrame.pack();
        viewFrame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent windowEvent) {
                if (!refreshing) {
                    closeWholePlugin();
                }
            }
        });
        viewFrame.setVisible(true);
    }
    
    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">                          
    private void initComponents() {

        startButton = new javax.swing.JButton();
        stopButton = new javax.swing.JButton();
        queuingPanel = new javax.swing.JPanel();
        queuingLabel = new javax.swing.JLabel();
        sendingPanel = new javax.swing.JPanel();
        sendingLabel = new javax.swing.JLabel();
        fpsLabel = new javax.swing.JLabel();
        serverTrace = new java.awt.TextArea();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setTitle("Camera Server");
        setSize(new java.awt.Dimension(500, 500));
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                formWindowClosing(evt);
            }
        });

        startButton.setFont(new java.awt.Font("Tahoma", 0, 18)); // NOI18N
        startButton.setText("Start Acquisition");
        startButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                startButtonActionPerformed(evt);
            }
        });

        stopButton.setFont(new java.awt.Font("Tahoma", 0, 18)); // NOI18N
        stopButton.setText("Stop Acquisition");
        stopButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                stopButtonActionPerformed(evt);
            }
        });

        queuingPanel.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        queuingLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        queuingLabel.setText("Image Queuing");

        javax.swing.GroupLayout queuingPanelLayout = new javax.swing.GroupLayout(queuingPanel);
        queuingPanel.setLayout(queuingPanelLayout);
        queuingPanelLayout.setHorizontalGroup(
            queuingPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(queuingPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(queuingLabel, javax.swing.GroupLayout.DEFAULT_SIZE, 95, Short.MAX_VALUE)
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

        fpsLabel.setText("FPS: -");

        serverTrace.setEditable(false);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(serverTrace, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(startButton)
                        .addGap(18, 18, 18)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(queuingPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(sendingPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(fpsLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 18, Short.MAX_VALUE)
                        .addComponent(stopButton)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(startButton)
                            .addComponent(stopButton)))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(queuingPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(3, 3, 3)
                        .addComponent(fpsLabel)
                        .addGap(1, 1, 1)
                        .addComponent(sendingPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(serverTrace, javax.swing.GroupLayout.DEFAULT_SIZE, 335, Short.MAX_VALUE)
                .addContainerGap())
        );

        pack();
    }// </editor-fold>                        

    private void startButtonActionPerformed(java.awt.event.ActionEvent evt) {                                            
        cc.startAcquisition();
    }                                           

    private void stopButtonActionPerformed(java.awt.event.ActionEvent evt) {                                           
        cc.stopAcquisition();
    }                                          
    
    private void formWindowClosing(java.awt.event.WindowEvent evt) {                                   
        closeWholePlugin();
    }                                  

    // Variables declaration - do not modify                     
    private javax.swing.JLabel fpsLabel;
    private javax.swing.JLabel queuingLabel;
    private javax.swing.JPanel queuingPanel;
    private javax.swing.JLabel sendingLabel;
    private javax.swing.JPanel sendingPanel;
    private java.awt.TextArea serverTrace;
    javax.swing.JButton startButton;
    javax.swing.JButton stopButton;
    // End of variables declaration                   

    @Override
    public void showText(String text) {
        serverTrace.append(text + "\n");
    }
}
