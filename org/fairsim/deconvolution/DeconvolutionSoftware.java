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

package org.fairsim.deconvolution;

import ij.ImagePlus;
import ij.io.FileSaver;
import ij.plugin.ChannelSplitter;
import ij.plugin.RGBStackMerge;
import ij.process.ImageProcessor;
import java.awt.Image;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.Timer;
import org.fairsim.linalg.*;
import org.fairsim.registration.Registration;

/**
 *
 * @author mpeplonski
 */
public class DeconvolutionSoftware extends javax.swing.JFrame {

    Registration reg;

    /**
     * Creates new form NewJFrame
     */
    public DeconvolutionSoftware() {
        initComponents();

        try {
            reg = new Registration(jTfTransformationMatrix.getText());
        } catch (IOException ex) {
            Logger.getLogger(DeconvolutionSoftware.class.getName()).log(Level.SEVERE, null, ex);
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

        buttonGroup1 = new javax.swing.ButtonGroup();
        BgMode = new javax.swing.ButtonGroup();
        jBtChooseFileImageStack = new javax.swing.JButton();
        jTfImageStack = new javax.swing.JTextField();
        jLbImageViewer = new javax.swing.JLabel();
        jBtSaveAlignedStack = new javax.swing.JButton();
        jSlSlice = new javax.swing.JSlider();
        jRb510 = new javax.swing.JRadioButton();
        jRb600 = new javax.swing.JRadioButton();
        jLbSlice = new javax.swing.JLabel();
        jLbImageStack = new javax.swing.JLabel();
        jRbManualMode = new javax.swing.JRadioButton();
        jRbLiveMode = new javax.swing.JRadioButton();
        jLbTransformationMatrix = new javax.swing.JLabel();
        jTfTransformationMatrix = new javax.swing.JTextField();
        jBtChooseFileTransformationMatrix = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        jTaLog = new javax.swing.JTextArea();
        jCbAlignImages = new javax.swing.JCheckBox();
        jBtGenerateTransformationMatrix = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jBtChooseFileImageStack.setText("choose file...");
        jBtChooseFileImageStack.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jBtChooseFileImageStackActionPerformed(evt);
            }
        });

        jLbImageViewer.setBackground(new java.awt.Color(255, 0, 0));
        jLbImageViewer.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        jBtSaveAlignedStack.setText("save aligned stack...");
        jBtSaveAlignedStack.setEnabled(false);
        jBtSaveAlignedStack.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jBtSaveAlignedStackActionPerformed(evt);
            }
        });

        jSlSlice.setMinimum(1);
        jSlSlice.setValue(0);
        jSlSlice.setEnabled(false);
        jSlSlice.addChangeListener(new javax.swing.event.ChangeListener() {
            public void stateChanged(javax.swing.event.ChangeEvent evt) {
                jSlSliceStateChanged(evt);
            }
        });

        buttonGroup1.add(jRb510);
        jRb510.setSelected(true);
        jRb510.setText("510 nm");
        jRb510.setEnabled(false);
        jRb510.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRb510ActionPerformed(evt);
            }
        });

        buttonGroup1.add(jRb600);
        jRb600.setText("600 nm");
        jRb600.setEnabled(false);
        jRb600.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRb600ActionPerformed(evt);
            }
        });

        jLbSlice.setHorizontalAlignment(javax.swing.SwingConstants.RIGHT);
        jLbSlice.setText("0 / 0");
        jLbSlice.setEnabled(false);

        jLbImageStack.setText("image stack:");

        BgMode.add(jRbManualMode);
        jRbManualMode.setSelected(true);
        jRbManualMode.setText("manual mode");
        jRbManualMode.setEnabled(false);
        jRbManualMode.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRbManualModeActionPerformed(evt);
            }
        });

        BgMode.add(jRbLiveMode);
        jRbLiveMode.setText("live mode");
        jRbLiveMode.setEnabled(false);
        jRbLiveMode.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jRbLiveModeActionPerformed(evt);
            }
        });

        jLbTransformationMatrix.setText("transformation matrix:");

        jTfTransformationMatrix.setText("C:\\usa_data\\transmatrix.txt");

        jBtChooseFileTransformationMatrix.setText("choose file...");
        jBtChooseFileTransformationMatrix.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jBtChooseFileTransformationMatrixActionPerformed(evt);
            }
        });

        jTaLog.setColumns(20);
        jTaLog.setRows(5);
        jScrollPane1.setViewportView(jTaLog);

        jCbAlignImages.setText("align images");
        jCbAlignImages.setEnabled(false);
        jCbAlignImages.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jCbAlignImagesActionPerformed(evt);
            }
        });

        jBtGenerateTransformationMatrix.setText("generate transformation matrix...");
        jBtGenerateTransformationMatrix.setEnabled(false);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jLbImageViewer, javax.swing.GroupLayout.PREFERRED_SIZE, 512, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                            .addComponent(jRbLiveMode)
                                            .addComponent(jRbManualMode))
                                        .addGap(61, 61, 61))
                                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                                        .addComponent(jRb510)
                                        .addGap(18, 18, 18)
                                        .addComponent(jRb600)
                                        .addGap(45, 45, 45))))
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(layout.createSequentialGroup()
                                        .addGap(6, 6, 6)
                                        .addComponent(jCbAlignImages)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                        .addComponent(jLbSlice, javax.swing.GroupLayout.PREFERRED_SIZE, 72, javax.swing.GroupLayout.PREFERRED_SIZE))
                                    .addGroup(layout.createSequentialGroup()
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                            .addComponent(jBtSaveAlignedStack, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                            .addComponent(jBtGenerateTransformationMatrix, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                            .addComponent(jSlSlice, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.DEFAULT_SIZE, 219, Short.MAX_VALUE))))
                                .addContainerGap())))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                    .addComponent(jLbTransformationMatrix)
                                    .addComponent(jLbImageStack))
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(layout.createSequentialGroup()
                                        .addComponent(jTfImageStack, javax.swing.GroupLayout.PREFERRED_SIZE, 501, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(jBtChooseFileImageStack))
                                    .addGroup(layout.createSequentialGroup()
                                        .addComponent(jTfTransformationMatrix, javax.swing.GroupLayout.PREFERRED_SIZE, 501, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(jBtChooseFileTransformationMatrix)))
                                .addGap(0, 0, Short.MAX_VALUE))
                            .addComponent(jScrollPane1))
                        .addContainerGap())))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(5, 5, 5)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLbImageStack)
                    .addComponent(jTfImageStack, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jBtChooseFileImageStack))
                .addGap(3, 3, 3)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLbTransformationMatrix)
                    .addComponent(jTfTransformationMatrix, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jBtChooseFileTransformationMatrix))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(jLbImageViewer, javax.swing.GroupLayout.PREFERRED_SIZE, 512, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(jRbManualMode)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jRbLiveMode)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(jBtGenerateTransformationMatrix)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jBtSaveAlignedStack)
                        .addGap(31, 31, 31)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(jRb510)
                            .addComponent(jRb600))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(jSlSlice, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(jCbAlignImages)
                            .addComponent(jLbSlice, javax.swing.GroupLayout.PREFERRED_SIZE, 22, javax.swing.GroupLayout.PREFERRED_SIZE))))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 121, Short.MAX_VALUE)
                .addContainerGap())
        );

        pack();
        setLocationRelativeTo(null);
    }// </editor-fold>//GEN-END:initComponents


    private void jBtChooseFileImageStackActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtChooseFileImageStackActionPerformed
        final JFileChooser fc = new JFileChooser();
        fc.setCurrentDirectory(new File("C:\\usa_data"));
        int returnVal = fc.showOpenDialog(null);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            jTfImageStack.setText(fc.getSelectedFile().getAbsolutePath());
            int stackSize = getStackSize();

            //Slider
            jSlSlice.setMaximum(stackSize);
            jSlSlice.setValue(1);
            jLbSlice.setText("1 / " + stackSize);
            jSlSlice.setEnabled(true);
            jCbAlignImages.setEnabled(true);
            jLbSlice.setEnabled(true);
            jBtSaveAlignedStack.setEnabled(true);
            jRbManualMode.setEnabled(true);
            jRb510.setEnabled(true);
            jRb600.setEnabled(true);

            loadImageToLabel();

            addLogMessage("Successfully loaded Image Stack '" + fc.getSelectedFile().getName() + "'");
        }
    }//GEN-LAST:event_jBtChooseFileImageStackActionPerformed

    private int getStackSize() {
        ImagePlus imgPlus = new ImagePlus(jTfImageStack.getText());
        ChannelSplitter cs = new ChannelSplitter();
        ImagePlus[] imgPlusSplitted = cs.split(imgPlus);
        return imgPlusSplitted[0].getStackSize();
    }

    private void loadImageToLabel() {
        ImagePlus imgPlus = new ImagePlus(jTfImageStack.getText());
        ChannelSplitter cs = new ChannelSplitter();
        ImagePlus[] imgPlusSplitted = cs.split(imgPlus);

        int stack;
        if (jRb510.isSelected()) {
            stack = 0;
        } else {
            stack = 1;
        }

        int slice;
        slice = jSlSlice.getValue();

        imgPlusSplitted[stack].setSlice(slice);

        Image image = imgPlusSplitted[stack].getImage();
        if (jCbAlignImages.isSelected()) {
            if (stack == 0) {
                image = alignImage(image, false, false);
            } else {
                image = alignImage(image, true, true);
            }
        }
        ImageIcon imageIcon = new ImageIcon(image);

        jLbImageViewer.setIcon(imageIcon);
    }

    private void addLogMessage(String message) {
        Calendar cal = Calendar.getInstance();
        SimpleDateFormat sdf = new SimpleDateFormat("HH:mm:ss");
        String timestamp = "[" + sdf.format(cal.getTime()) + "] ";
        jTaLog.setText(timestamp + message + "\r\n" + jTaLog.getText());
        jTaLog.setCaretPosition(0);
    }

    private void jBtSaveAlignedStackActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtSaveAlignedStackActionPerformed
        JFileChooser fc = new JFileChooser();
        fc.setCurrentDirectory(new File("C:\\usa_data"));
        int returnVal = fc.showSaveDialog(null);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            File fileToSave = fc.getSelectedFile();

            ImagePlus imgPlus = new ImagePlus(jTfImageStack.getText());
            ChannelSplitter cs = new ChannelSplitter();
            ImagePlus[] imgPlusSplitted = cs.split(imgPlus);
            
            
            
            
            

            // flip images horizontally
            for (int i = 1; i <= imgPlusSplitted[1].getStackSize(); i++) {
                imgPlusSplitted[1].setSlice(i);
                ImageProcessor imgProcessor = imgPlusSplitted[1].getProcessor();
                imgProcessor.flipHorizontal();
            }

            // registration
            Registration.setRecon(true);

            for (int i = 1; i <= imgPlusSplitted[1].getStackSize(); i++) {
                imgPlusSplitted[1].setSlice(i);

                ImageProcessor imgProcessor = imgPlusSplitted[1].getProcessor();
                VectorFactory vf = BasicVectors.getFactory();
                Vec2d.Real source = vf.createReal2D(imgProcessor.getWidth(), imgProcessor.getHeight());

                for (int y = 0; y < imgProcessor.getHeight(); y++) {
                    for (int x = 0; x < imgProcessor.getWidth(); x++) {
                        source.set(x, y, imgProcessor.get(x, y));
                    }
                }

                if (Registration.isRecon()) {
                    Vec2d.Real regVec = reg.registerReconImage(source);

                    for (int y = 0; y < imgProcessor.getHeight(); y++) {
                        for (int x = 0; x < imgProcessor.getWidth(); x++) {
                            imgProcessor.set(x, y, (int) regVec.get(x, y));
                        }
                    }
                }

            }

            RGBStackMerge merge = new RGBStackMerge();
            ImagePlus result = merge.mergeChannels(imgPlusSplitted, true);

            FileSaver fs = new FileSaver(result);
            fs.saveAsTiff(fileToSave.getAbsolutePath());
            
            addLogMessage("Image Stack successfully saved to '" + fileToSave.getAbsolutePath() + "'");
        }
    }//GEN-LAST:event_jBtSaveAlignedStackActionPerformed

    private Image alignImage(Image image, boolean applyRegistration, boolean applyHorizontallyFlip) {
        ImagePlus imgPlus = new ImagePlus("", image);
        ImageProcessor imgProcessor = imgPlus.getProcessor();

        if (applyHorizontallyFlip) {
            imgProcessor.flipHorizontal();
        }

        if (applyRegistration) {
            Registration.setRecon(true);

            VectorFactory vf = BasicVectors.getFactory();
            Vec2d.Real source = vf.createReal2D(imgProcessor.getWidth(), imgProcessor.getHeight());

            for (int y = 0; y < imgProcessor.getHeight(); y++) {
                for (int x = 0; x < imgProcessor.getWidth(); x++) {
                    source.set(x, y, imgProcessor.get(x, y));
                }
            }

            if (Registration.isRecon()) {
                Vec2d.Real regVec = reg.registerReconImage(source);

                for (int y = 0; y < imgProcessor.getHeight(); y++) {
                    for (int x = 0; x < imgProcessor.getWidth(); x++) {
                        imgProcessor.set(x, y, (int) regVec.get(x, y));
                    }
                }
            }
            Registration.setRecon(false);
        }

        return imgPlus.getImage();
    }

    private javax.swing.Timer timer;

    private String getFileroot(String fullPath) {
        File f = new File(fullPath);
        return f.getName().substring(0, f.getName().lastIndexOf("."));
    }

    private String getFileextension(String fullPath) {
        File f = new File(fullPath);
        return f.getName().substring(f.getName().lastIndexOf(".") + 1);
    }

    private String getDirname(String fullPath) {
        File f = new File(fullPath);
        return f.getParent();
    }

    private void jSlSliceStateChanged(javax.swing.event.ChangeEvent evt) {//GEN-FIRST:event_jSlSliceStateChanged
        if (timer != null) {
            timer.stop();
        }
        timer = new Timer(50, new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                jLbSlice.setText(jSlSlice.getValue() + " / " + getStackSize());
                int stackNumber;
                if (jRb510.isSelected()) {
                    stackNumber = 0;
                } else {
                    stackNumber = 1;
                }
                loadImageToLabel();
                timer = null;
            }
        });
        timer.setRepeats(false);
        timer.start();

        /*
        jLbImageInStack.setText(jSlImageInStack.getValue() + " / " + getStackSize());
        int stackNumber;
        if(jRb510.isSelected()) {
            stackNumber = 0;
        }
        else {
            stackNumber = 1;
        }
        loadImageToLabel(stackNumber, jSlImageInStack.getValue());*/
    }//GEN-LAST:event_jSlSliceStateChanged

    private void jRb600ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRb600ActionPerformed
        loadImageToLabel();
    }//GEN-LAST:event_jRb600ActionPerformed

    private void jRb510ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRb510ActionPerformed
        loadImageToLabel();
    }//GEN-LAST:event_jRb510ActionPerformed

    private void jRbLiveModeActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRbLiveModeActionPerformed
        jTfImageStack.setEnabled(false);
        jBtChooseFileImageStack.setEnabled(false);
        addLogMessage("Live Mode activated");
    }//GEN-LAST:event_jRbLiveModeActionPerformed

    private void jRbManualModeActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jRbManualModeActionPerformed
        jTfImageStack.setEnabled(true);
        jBtChooseFileImageStack.setEnabled(true);
        addLogMessage("Manual Mode activated");
    }//GEN-LAST:event_jRbManualModeActionPerformed

    private void jBtChooseFileTransformationMatrixActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jBtChooseFileTransformationMatrixActionPerformed
        final JFileChooser fc = new JFileChooser();
        fc.setCurrentDirectory(new File("C:\\usa_data"));
        int returnVal = fc.showOpenDialog(null);
        if (returnVal == JFileChooser.APPROVE_OPTION) {
            jTfTransformationMatrix.setText(fc.getSelectedFile().getAbsolutePath());

            try {
                reg = new Registration(jTfTransformationMatrix.getText());
            } catch (IOException ex) {
                Logger.getLogger(DeconvolutionSoftware.class.getName()).log(Level.SEVERE, null, ex);
            }

            addLogMessage("Successfully loaded Transformation Matrix '" + fc.getSelectedFile().getName() + "'");
        }
    }//GEN-LAST:event_jBtChooseFileTransformationMatrixActionPerformed

    private void jCbAlignImagesActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jCbAlignImagesActionPerformed
        loadImageToLabel();
    }//GEN-LAST:event_jCbAlignImagesActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(DeconvolutionSoftware.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(DeconvolutionSoftware.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(DeconvolutionSoftware.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(DeconvolutionSoftware.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new DeconvolutionSoftware().setVisible(true);
            }
        });
    }


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.ButtonGroup BgMode;
    private javax.swing.ButtonGroup buttonGroup1;
    private javax.swing.JButton jBtChooseFileImageStack;
    private javax.swing.JButton jBtChooseFileTransformationMatrix;
    private javax.swing.JButton jBtGenerateTransformationMatrix;
    private javax.swing.JButton jBtSaveAlignedStack;
    private javax.swing.JCheckBox jCbAlignImages;
    private javax.swing.JLabel jLbImageStack;
    private javax.swing.JLabel jLbImageViewer;
    private javax.swing.JLabel jLbSlice;
    private javax.swing.JLabel jLbTransformationMatrix;
    private javax.swing.JRadioButton jRb510;
    private javax.swing.JRadioButton jRb600;
    private javax.swing.JRadioButton jRbLiveMode;
    private javax.swing.JRadioButton jRbManualMode;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JSlider jSlSlice;
    private javax.swing.JTextArea jTaLog;
    private javax.swing.JTextField jTfImageStack;
    private javax.swing.JTextField jTfTransformationMatrix;
    // End of variables declaration//GEN-END:variables
}