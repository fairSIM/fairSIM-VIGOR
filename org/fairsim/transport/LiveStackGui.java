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
package org.fairsim.transport;

import java.io.File;
import java.io.IOException;
import javax.swing.DefaultListModel;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.filechooser.FileFilter;
import javax.swing.filechooser.FileNameExtensionFilter;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.livemode.LiveControlPanel;

/**
 *
 * @author m.lachetta
 */
public class LiveStackGui extends javax.swing.JPanel {

    File lsDir;
    File tarDir;
    File[] files;
    VectorFactory vf;
    
    /**
     * Creates new form LiveStackGui
     */
    public LiveStackGui(String liveStackDirectory, VectorFactory vf) throws IOException {
        
        this.lsDir = new File(liveStackDirectory);
        this.vf = vf;
        if (!lsDir.isDirectory()) throw new IOException(liveStackDirectory + " is not a Directory");
        tarDir = lsDir;
        files = new File[0];
        
        initComponents();
    }
    
    private File openTargetDirectory() throws IOException {
        JFileChooser fc = new JFileChooser(lsDir);
        fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        int state = fc.showOpenDialog(this);
        if (state == JFileChooser.APPROVE_OPTION) {
            File dir = fc.getSelectedFile();
            if (dir.isDirectory()) return dir;
            else throw new IOException("Not a directory");
        } else throw new IOException("Directory selection failed");
    }
    
    private File[] openLiveStackFiles() throws IOException {
        JFileChooser fc = new JFileChooser(lsDir);
        FileFilter ff = new FileNameExtensionFilter("LiveStack files", "livestack", "livestack.tif");
        fc.setMultiSelectionEnabled(true);
        fc.setFileFilter(ff);
        
        
        int state = fc.showOpenDialog(this);
        if (state == JFileChooser.APPROVE_OPTION) {
            File[] files = fc.getSelectedFiles();
            if (files.length > 0) return files;
            else throw new IOException("No files selected");
        } else throw new IOException("File selection failed");
    }
    
    private void updateFileList() {
        DefaultListModel<String> model = (DefaultListModel<String>) fileList.getModel();
        model.clear();
        for (File f : files) {
            model.addElement(f.getName());
        }
    }
    
    private void printText(String text) {
        logTextArea.append(text + "\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
    }
    
    private void printError(String message) {
        logTextArea.append("Error: " + message + "\n");
        logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
    }
    
    private void converteFile(File f, boolean tiff, boolean meta, boolean rec) {
        if (!(tiff | meta | rec)) printError("No action selected");
        else {
            try {
                printText("Readin: " + f);
                LiveStack ls = new LiveStack(f.getAbsolutePath());
                String fileName = f.getName();
                if (!fileName.endsWith(".livestack.tif") && !fileName.endsWith(".livestack"))
                    throw new IOException("Expected .livestack or .livestack.tif file");
                if (fileName.endsWith(".livestack.tif"))
                    fileName = fileName.substring(0, fileName.length() - 4);
                if (tiff) {
                    printText("Save as Tiff: " + fileName + ".tif");
                    ls.saveAsTiff(tarDir + "/" + fileName + ".tif");
                }
                if (meta) {
                    printText("Save meta file: " + fileName + "---.meta.txt");
                    ls.toMeta(tarDir + "/" + fileName);
                }
                if (rec) {
                    printText("Reconstruct stack by header parameters");
                    LiveStack.SimReconstructor sr = ls.loadSimReconstructor(vf);
                    LiveStack.ReconStack rs = ls.reconstructByHeader(sr);
                    printText("Save widefield: " + tarDir + "/" + fileName + ".wf.tif");
                    rs.saveWfAsTiff(tarDir + "/" + fileName + ".wf.tif");
                    printText("Save reconstruction: " + tarDir + "/" + fileName + ".recon.tif");
                    rs.saveReconAsTiff(tarDir + "/" + fileName + ".recon.tif");
                    printText("Finished: " + f);
                }
            } catch (Exception ex) {
                //ex.printStackTrace();
                printError("Converting failed: " + f + " " + ex);
            }
        }
    }
    
    private void setButtonsEnabled(boolean b) {
        openButton.setEnabled(b);
        changeButton.setEnabled(b);
        convertSelectedButton.setEnabled(b);
        convertAllButton.setEnabled(b);
    }
    
    public static void main(String[] args) throws IOException {
        LiveStackGui lsg = new LiveStackGui("G:\\downloads", LiveControlPanel.loadVectorFactory());
        JFrame testFrame = new JFrame("LiveStackGui test");
        testFrame.add(lsg);
        testFrame.pack();
        testFrame.setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        testFrame.setVisible(true);
//        File[] files = lsg.openFiles();
//        System.out.println(files.length);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        logPanel = new javax.swing.JPanel();
        logScrollPane = new javax.swing.JScrollPane();
        logTextArea = new javax.swing.JTextArea();
        converterPanel = new javax.swing.JPanel();
        openButton = new javax.swing.JButton();
        fileListScrollPane = new javax.swing.JScrollPane();
        fileList = new javax.swing.JList<>();
        tiffCheckBox = new javax.swing.JCheckBox();
        metaCheckBox = new javax.swing.JCheckBox();
        recCheckBox = new javax.swing.JCheckBox();
        targetLabel = new javax.swing.JLabel();
        targetTextField = new javax.swing.JTextField();
        changeButton = new javax.swing.JButton();
        convertSelectedButton = new javax.swing.JButton();
        metaScrollPane = new javax.swing.JScrollPane();
        metaTextArea = new javax.swing.JTextArea();
        convertAllButton = new javax.swing.JButton();

        logPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("Log"));

        logTextArea.setEditable(false);
        logTextArea.setRows(10);
        logScrollPane.setViewportView(logTextArea);

        javax.swing.GroupLayout logPanelLayout = new javax.swing.GroupLayout(logPanel);
        logPanel.setLayout(logPanelLayout);
        logPanelLayout.setHorizontalGroup(
            logPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(logPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(logScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 0, Short.MAX_VALUE)
                .addContainerGap())
        );
        logPanelLayout.setVerticalGroup(
            logPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(logPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(logScrollPane)
                .addContainerGap())
        );

        converterPanel.setBorder(javax.swing.BorderFactory.createTitledBorder("livestack converter"));

        openButton.setText("open");
        openButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                openButtonActionPerformed(evt);
            }
        });

        fileList.setModel(new DefaultListModel<String>());
        fileList.setSelectionMode(javax.swing.ListSelectionModel.SINGLE_SELECTION);
        fileList.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseClicked(java.awt.event.MouseEvent evt) {
                fileListMouseClicked(evt);
            }
        });
        fileListScrollPane.setViewportView(fileList);

        tiffCheckBox.setText("save as tiff");

        metaCheckBox.setText("save meta file");

        recCheckBox.setText("reconstruct");
        recCheckBox.setToolTipText("");

        targetLabel.setText("target directory:");

        targetTextField.setEditable(false);
        targetTextField.setText(lsDir.getAbsolutePath());

        changeButton.setText("change");
        changeButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                changeButtonActionPerformed(evt);
            }
        });

        convertSelectedButton.setText("convert selected");
        convertSelectedButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                convertSelectedButtonActionPerformed(evt);
            }
        });

        metaTextArea.setEditable(false);
        metaTextArea.setColumns(20);
        metaTextArea.setRows(5);
        metaScrollPane.setViewportView(metaTextArea);

        convertAllButton.setText("converte whole list");
        convertAllButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                convertAllButtonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout converterPanelLayout = new javax.swing.GroupLayout(converterPanel);
        converterPanel.setLayout(converterPanelLayout);
        converterPanelLayout.setHorizontalGroup(
            converterPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(converterPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(converterPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(converterPanelLayout.createSequentialGroup()
                        .addGroup(converterPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(openButton)
                            .addComponent(fileListScrollPane, javax.swing.GroupLayout.PREFERRED_SIZE, 289, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(metaScrollPane))
                    .addGroup(converterPanelLayout.createSequentialGroup()
                        .addComponent(targetLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(targetTextField)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(changeButton))
                    .addGroup(converterPanelLayout.createSequentialGroup()
                        .addGroup(converterPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(converterPanelLayout.createSequentialGroup()
                                .addComponent(convertSelectedButton)
                                .addGap(18, 18, 18)
                                .addComponent(convertAllButton))
                            .addGroup(converterPanelLayout.createSequentialGroup()
                                .addComponent(tiffCheckBox)
                                .addGap(18, 18, 18)
                                .addComponent(metaCheckBox)
                                .addGap(18, 18, 18)
                                .addComponent(recCheckBox)))
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );
        converterPanelLayout.setVerticalGroup(
            converterPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(converterPanelLayout.createSequentialGroup()
                .addContainerGap()
                .addComponent(openButton)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(converterPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(metaScrollPane, javax.swing.GroupLayout.DEFAULT_SIZE, 43, Short.MAX_VALUE)
                    .addComponent(fileListScrollPane, javax.swing.GroupLayout.DEFAULT_SIZE, 43, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(converterPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(tiffCheckBox)
                    .addComponent(metaCheckBox)
                    .addComponent(recCheckBox))
                .addGap(18, 18, 18)
                .addGroup(converterPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(targetLabel)
                    .addComponent(targetTextField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(changeButton))
                .addGap(18, 18, 18)
                .addGroup(converterPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(convertSelectedButton)
                    .addComponent(convertAllButton))
                .addContainerGap())
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(converterPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(logPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(converterPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(logPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    private void openButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_openButtonActionPerformed
        try {
            files = openLiveStackFiles();
            printText("Opened " + files.length + " files");
        } catch (IOException ex) {
            files = new File[0];
            printError(ex.getMessage());
        }
        updateFileList();
    }//GEN-LAST:event_openButtonActionPerformed

    private void changeButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_changeButtonActionPerformed
        try {
            tarDir = openTargetDirectory();
            targetTextField.setText(tarDir.getAbsolutePath());
            printText("Set target directory to: " + tarDir);
        } catch (IOException ex) {
            printError(ex.getMessage());
        }
    }//GEN-LAST:event_changeButtonActionPerformed

    private void fileListMouseClicked(java.awt.event.MouseEvent evt) {//GEN-FIRST:event_fileListMouseClicked
        int fileIdx = fileList.getSelectedIndex();
        if (fileIdx >= 0) {
            if (fileIdx >= files.length) printError("Selected file not in list " + fileIdx);
            else {
                String filePath = files[fileIdx].getAbsolutePath();
                String header;
                try {
                    header = LiveStack.readHeaderFromFile(filePath).getStringRepresentation();
                } catch (IOException ex) {
                    header = "header preview only supported \nfor .livestack files";
                }
                metaTextArea.setText(header);
                metaTextArea.setCaretPosition(0);
            }
        }
    }//GEN-LAST:event_fileListMouseClicked

    private void convertSelectedButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_convertSelectedButtonActionPerformed
        int fileIdx = fileList.getSelectedIndex();
        if (fileIdx >= 0) {
            if (fileIdx >= files.length) printError("Selected file not in list " + fileIdx);
            else {
                File f = files[fileIdx];
                setButtonsEnabled(false);
                new Thread(() -> {
                    converteFile(f, tiffCheckBox.isSelected(), metaCheckBox.isSelected(), recCheckBox.isSelected());
                    setButtonsEnabled(true);
                }).start();
                
            }
        } else if (fileIdx < 0) printError("No file selected");
        else printError("Unknown file selection error: " + fileIdx);
    }//GEN-LAST:event_convertSelectedButtonActionPerformed

    private void convertAllButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_convertAllButtonActionPerformed
        if (files.length > 0) {
            setButtonsEnabled(false);
            boolean tiff = tiffCheckBox.isSelected();
            boolean meta = metaCheckBox.isSelected();
            boolean rec = recCheckBox.isSelected();
            new Thread(() -> {
                for (File f : files) {
                    converteFile(f, tiff, meta, rec);
                }
                setButtonsEnabled(true);
            }).start();
        } else printError("File list is empty");
    }//GEN-LAST:event_convertAllButtonActionPerformed


    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton changeButton;
    private javax.swing.JButton convertAllButton;
    private javax.swing.JButton convertSelectedButton;
    private javax.swing.JPanel converterPanel;
    private javax.swing.JList<String> fileList;
    private javax.swing.JScrollPane fileListScrollPane;
    private javax.swing.JPanel logPanel;
    private javax.swing.JScrollPane logScrollPane;
    private javax.swing.JTextArea logTextArea;
    private javax.swing.JCheckBox metaCheckBox;
    private javax.swing.JScrollPane metaScrollPane;
    private javax.swing.JTextArea metaTextArea;
    private javax.swing.JButton openButton;
    private javax.swing.JCheckBox recCheckBox;
    private javax.swing.JLabel targetLabel;
    private javax.swing.JTextField targetTextField;
    private javax.swing.JCheckBox tiffCheckBox;
    // End of variables declaration//GEN-END:variables
}
