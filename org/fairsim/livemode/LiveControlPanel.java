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
package org.fairsim.livemode;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JButton;
import javax.swing.BoxLayout;
import javax.swing.BorderFactory;
import javax.swing.JProgressBar;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.text.DefaultCaret;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;

import java.awt.Color;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.GridLayout;

import java.util.Arrays;
import java.util.concurrent.BlockingQueue;
import javax.swing.Box;
import javax.swing.JLabel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.fairsim.utils.Conf;
import org.fairsim.transport.ImageReceiver;
import org.fairsim.transport.ImageDiskWriter;
import org.fairsim.linalg.VectorFactory;
import org.fairsim.accel.AccelVectorFactory;
import org.fairsim.linalg.Vec;
import org.fairsim.sim_gui.PlainImageDisplay;

import org.fairsim.linalg.Vec2d;
import org.fairsim.controller.AdvancedGui;
import org.fairsim.controller.EasyGui;
import org.fairsim.utils.Tool;
import org.fairsim.utils.SimpleMT;

/**
 * Provides the control interface for live mode
 */
public class LiveControlPanel {

    boolean isRecording = false;    // if the raw stream is recorded

    int wfPixelSize;
    private final int nrCh;
    private final String[] channels;

    final JProgressBar networkBufferBar;
    final JProgressBar reconBufferInputBar;
    final JProgressBar reconBufferOutputBar;
    final JProgressBar[] channelBufferBar;

    final JProgressBar fileBufferBar;

    // The different threads in use:
    final ImageDiskWriter liveStreamWriter;
    final ImageReceiver imageReceiver;
    final ReconstructionRunner reconRunner;
    final SimSequenceExtractor seqDetection;

    

    final JTextArea statusField;
    final JTextField statusMessage;
    ParameterTab[] pTab;
    JButton[] syncButtons;
    
    private JFrame hrFr;
    private JFrame lrFr;
    PlainImageDisplay wfDisplay;
    PlainImageDisplay reconDisplay;
    
    SimpleImageForward sif1;
    SimpleImageForward sif2;
    DynamicDisplayUpdate updateThread;

    public LiveControlPanel(final Conf.Folder cfg,
            VectorFactory avf, String[] channels)
            throws Conf.EntryNotFoundException, java.io.IOException {
        
        this.channels = channels;
        nrCh = channels.length;

        // get parameters
        //final int imgSize = cfg.getInt("NetworkBuffer").val();

        //  ------- 
        //  initialize the GUI
        //  ------- 
        JFrame mainFrame = new JFrame("Live SIM control");
        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel, BoxLayout.Y_AXIS));

        // GUI - buffering
        JPanel reconBuffersPanel = new JPanel();
        reconBuffersPanel.setBorder(BorderFactory.createTitledBorder(
                "Reconstruction buffers"));
        reconBuffersPanel.setLayout(new GridLayout(3 + nrCh, 1, 2, 2));

        networkBufferBar = new JProgressBar();
        networkBufferBar.setString("network input buffer: 0%");
        networkBufferBar.setStringPainted(true);

        reconBufferInputBar = new JProgressBar();
        reconBufferInputBar.setString("recon input buffer: 0%");
        reconBufferInputBar.setStringPainted(true);

        reconBufferOutputBar = new JProgressBar();
        reconBufferOutputBar.setString("recon output buffer: 0%");
        reconBufferOutputBar.setStringPainted(true);

        reconBuffersPanel.add(networkBufferBar);
        reconBuffersPanel.add(reconBufferInputBar);
        reconBuffersPanel.add(reconBufferOutputBar);

        mainPanel.add(reconBuffersPanel);
        
        // GUI - image record function
        JPanel recorderPanel = new JPanel();
        recorderPanel.setBorder(BorderFactory.createTitledBorder(
                "raw Stream recording"));
        recorderPanel.setLayout(new GridLayout(3, 1, 2, 2));

        final JTextField filePrefix = new JTextField("VIGOR", 30);

        final JButton recordButton = new JButton("record");
        recordButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                if (!isRecording) {
                    recordButton.setForeground(Color.RED);
                    try {
                        liveStreamWriter.startRecording(filePrefix.getText());
                    } catch (Exception ex) {
                        throw new RuntimeException(ex);
                    }
                    isRecording = true;
                } else {
                    recordButton.setForeground(Color.BLACK);
                    liveStreamWriter.stopRecording();
                    isRecording = false;
                }
            }
        ;
        });

	final JButton bufferClearButton = new JButton("buffer clear / resync");
        bufferClearButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                seqDetection.clearBuffers();
            }
        ;
        });


	fileBufferBar = new JProgressBar();
        fileBufferBar.setString("save buffer");
        fileBufferBar.setStringPainted(true);

        recorderPanel.add(recordButton);
        recorderPanel.add(bufferClearButton);
        recorderPanel.add(filePrefix);
        recorderPanel.add(fileBufferBar);
        mainPanel.add(recorderPanel);

        //mainPanel.add(new RegistrationPanel(avf, cfg, channels));
        JButton fitPeakButton = new JButton("run parameter fit");
        fitPeakButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                //rt.triggerParamRefit();
            }
        ;
        });

	// error output and such
	JPanel statusPanel = new JPanel();
        statusPanel.setBorder(BorderFactory.createTitledBorder(
                "status messages"));
        statusField = new JTextArea(30, 60);
        statusField.setEditable(false);
        DefaultCaret cr = (DefaultCaret) statusField.getCaret();
        cr.setUpdatePolicy(DefaultCaret.ALWAYS_UPDATE);
        JScrollPane statusScroller = new JScrollPane(statusField);
        statusPanel.add(statusScroller);
        mainPanel.add(statusPanel);

        wfPixelSize = cfg.getInt("RawPxlCount").val();

        JPanel statusPanel2 = new JPanel();
        syncButtons = new JButton[nrCh];
        channelBufferBar = new JProgressBar[nrCh];
        for (int c = 0; c < nrCh; c++) {
            syncButtons[c] = new JButton("");
            syncButtons[c].setEnabled(false);
            statusPanel2.add(syncButtons[c]);
            channelBufferBar[c] = new JProgressBar();
            channelBufferBar[c].setString(channels[c] + " channel buffer: 0%");
            channelBufferBar[c].setStringPainted(true);
            reconBuffersPanel.add(channelBufferBar[c]);
        }

        statusMessage = new JTextField(40);
        statusMessage.setEditable(false);

        statusPanel2.add(statusMessage);

        mainPanel.add(statusPanel2);

        // redirect log output
        Tool.setLogger(new Tool.Logger() {
            public void writeTrace(String e) {
                statusField.append(e + "\n");
            }

            public void writeShortMessage(String e) {
                statusMessage.setText(e);
            }

            public void writeError(String e, boolean fatal) {
                statusField.append((fatal) ? ("FATAL err: ") : ("ERR :: ") + e + "\n");
            }
        });

        //  ------- 
        //  initialize the components
        //  ------- 
        // network receiver and image storage
        int netBufferSize = cfg.getInt("NetworkBuffer").val();
        imageReceiver = new ImageReceiver(netBufferSize, wfPixelSize, wfPixelSize);

        String saveFolder = cfg.getStr("DiskFolder").val();
        int diskBufferSize = cfg.getInt("DiskBuffer").val();
        liveStreamWriter = new ImageDiskWriter(saveFolder, diskBufferSize);
        imageReceiver.setDiskWriter(liveStreamWriter);

        // start the network receiver
        imageReceiver.startReceiving(null, null);

        // start the reconstruction threads
        reconRunner = new ReconstructionRunner(cfg, avf, channels);

        // start the SIM sequence detection
        seqDetection = new SimSequenceExtractor(cfg, imageReceiver, reconRunner, this);

        // setup the displays
        initView();

        // setup main interface tabs
        JTabbedPane tabbedPane = new JTabbedPane();

        tabbedPane.addTab("main", mainPanel);
        AdvancedGui advancedGui = new AdvancedGui(cfg, channels, this);
        EasyGui easyGui = new EasyGui(advancedGui);
        tabbedPane.addTab("Easy", easyGui);
        tabbedPane.addTab("Advanced", advancedGui);

        pTab = new ParameterTab[nrCh];
        for (int ch = 0; ch < nrCh; ch++) {
            pTab[ch] = new ParameterTab(reconRunner, ch, cfg);
            tabbedPane.addTab(channels[ch], pTab[ch].getPanel());
        }

        mainFrame.add(tabbedPane);
        mainFrame.pack();
        mainFrame.setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        mainFrame.setVisible(true);

        startThreads();

    }
    
    private void initView() {
        if (reconRunner != null) wfDisplay = new RawImageDisplay(this.reconRunner, channels);
        else wfDisplay = new PlainImageDisplay(nrCh, wfPixelSize, wfPixelSize, channels);
        reconDisplay = new PlainImageDisplay(nrCh, 2 * wfPixelSize, 2 * wfPixelSize, channels);
        hrFr = new JFrame("Reconstruction");
        lrFr = new JFrame("Widefiled");
        hrFr.add(reconDisplay.getPanel());
        lrFr.add(wfDisplay.getPanel());
        hrFr.pack();
        lrFr.pack();

        hrFr.setVisible(true);
        lrFr.setVisible(true);
    }
    
    public void refreshView(int pixelSize) {
        stopThreads();
        seqDetection.pause(true);
        this.wfPixelSize = pixelSize;
        imageReceiver.setImageSize(wfPixelSize, wfPixelSize);
        seqDetection.clearBuffers();
        seqDetection.resetChannelBufferThreads();
        reconRunner.setImageSize(wfPixelSize);
        
        hrFr.dispose();
        lrFr.dispose();
        wfDisplay = null;
        reconDisplay = null;
        initView();
        hrFr.toFront();
        lrFr.toFront();
        seqDetection.pause(false);
        startThreads();
    }
    
    private void startThreads() {
        sif1 = new SimpleImageForward(false);
        sif2 = new SimpleImageForward(true);

        sif1.start();
        sif2.start();

        updateThread = new DynamicDisplayUpdate();
        updateThread.start();
        Tool.trace("Started DisplayThreads.");
    }
    
    private void stopThreads() {
        sif1.interrupt();
        sif2.interrupt();
        updateThread.interrupt();
        try {
            sif1.join();
            sif2.join();
            updateThread.join();
        } catch (InterruptedException ex) {
            throw new RuntimeException(ex);
        }
        Tool.trace("Stopped DisplayThreads.");
    }
    
    public int getWfSize() {
        return this.wfPixelSize;
    }
    
    public ReconstructionRunner getReconRunner() {
        return this.reconRunner;
    }
    
    public SimSequenceExtractor getSequenceExtractor() {
        return this.seqDetection;
    }

    /**
     * Thread updating dynamic display
     */
    class DynamicDisplayUpdate extends Thread {
        
        private int getQueuePercent(BlockingQueue queue) {
            int used = queue.size();
            int free = queue.remainingCapacity();
            return used * 100 / (used + free);
        }

        public void run() {
            while (true) {
                int bufferPercent = imageReceiver.getQueuePercent();
                networkBufferBar.setString("network input buffer: " + bufferPercent + '%');
                networkBufferBar.setValue(bufferPercent);
                bufferPercent = getQueuePercent(reconRunner.imgsToReconstruct);
                reconBufferInputBar.setString("recon input buffer: " + bufferPercent + '%');
                reconBufferInputBar.setValue(bufferPercent);
                bufferPercent = getQueuePercent(reconRunner.finalRecon);
                reconBufferOutputBar.setString("recon output buffer: " + bufferPercent + '%');
                reconBufferOutputBar.setValue(bufferPercent);
                for(int ch = 0; ch < nrCh; ch++) {
                    int percent = getQueuePercent(seqDetection.channels[ch].rawImgs);
                    channelBufferBar[ch].setString(channels[ch] + " channel buffer: " + percent + "%");
                    channelBufferBar[ch].setValue(percent);
                }
                // update save buffer state
                fileBufferBar.setString(String.format("%7.0f MB / %7.0f sec left",
                        (float) liveStreamWriter.getSpace() / (2*wfPixelSize) / (2*wfPixelSize),
                        liveStreamWriter.getTimeLeft(wfPixelSize, 1, 100)));
                fileBufferBar.setValue(liveStreamWriter.bufferState());

                int dropped = liveStreamWriter.nrDroppedFrames();
                if (dropped > 0 && isRecording) {
                    Tool.error("#" + dropped + " not saved", false);
                }

                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    return;
                }
            }
        }

    }

    class SimpleImageForward extends Thread {

        final boolean doWidefield;

        SimpleImageForward(boolean dwf) {
            doWidefield = dwf;
        }

        public void run() {
            while (!isInterrupted()) {
                try {
                    Vec2d.Real[] img;
                    if (doWidefield) {
                        img = reconRunner.finalWidefield.take();
                    } else {
                        img = reconRunner.finalRecon.take();
                    }

                    for (int c = 0; c < reconRunner.nrChannels; c++) {
                        if (doWidefield) {
                            wfDisplay.newImage(c, img[c]);
                        } else {
                            reconDisplay.newImage(c, img[c]);
                        }
                    }
                } catch (InterruptedException e) {
                    Tool.trace("Display thread interrupted, why?");
                    interrupt();
                    continue;
                }

                if (doWidefield) {
                    wfDisplay.refresh();
                } else {
                    reconDisplay.refresh();
                }
            }
        }
    }
    
    public static class RawImageDisplay extends PlainImageDisplay {

        public RawImageDisplay(ReconstructionRunner recRunner, String... names) {
            super(recRunner.nrChannels, recRunner.width, recRunner.height, names);
            JLabel rawLabel = new JLabel("Raw Frame: " + recRunner.rawOutput);
            JSlider rawSlider = new JSlider(JSlider.HORIZONTAL, 0, recRunner.nrDirs * recRunner.nrPhases, 0);
            rawSlider.addChangeListener(new ChangeListener() {
                public void stateChanged(ChangeEvent e) {
                    recRunner.rawOutput = rawSlider.getValue();
                    rawLabel.setText("Raw Frame: " + recRunner.rawOutput);
                    refresh();
                }
            });

            JPanel p = new JPanel();
            p.setLayout(new BoxLayout(p, BoxLayout.X_AXIS));
            p.add(Box.createHorizontalStrut(20));
            p.add(rawLabel);
            p.add(Box.createHorizontalStrut(20));
            p.add(rawSlider);
            p.add(Box.createHorizontalStrut(20));

            mainPanel.add(p, 1);
            mainPanel.add(Box.createVerticalStrut(20), 1);

        }
    }

    /**
     * starts and displays the GUI
     */
    public static void main(String[] arg) {
        // tweak SimpleMT
        SimpleMT.setNrThreads(Math.max(SimpleMT.getNrThreads() - 2, 2));

        // load the CUDA library
        String OS = System.getProperty("os.name").toLowerCase();
        VectorFactory avf;

        // following Factory for Linux-GPU-Reconstruction
        if (OS.contains("nix") || OS.contains("nux") || OS.contains("aix")) {
            String wd = System.getProperty("user.dir") + "/accel/";
            Tool.trace("loading library from: " + wd);
            try {
                System.load(wd + "libcudaimpl.so");
                avf = AccelVectorFactory.getFactory();
            } catch (UnsatisfiedLinkError ex) {
                System.err.println("[fairSIM] Error: " + ex);
                System.err.println("[fairSIM] Error: now loading not GPU supported version");
                avf = Vec.getBasicVectorFactory();
            }
        } // following Factory for Windows-GPU-Reconstruction
        else if (OS.contains("win")) {
            String wd = System.getProperty("user.dir") + "\\";
            Tool.trace("loading library from: " + wd);
            try {
                System.load(wd + "libcudaimpl.dll");
                avf = AccelVectorFactory.getFactory();
            } catch (UnsatisfiedLinkError ex) {
                System.err.println("[fairSIM] Error: " + ex);
                System.err.println("[fairSIM] Error: now loading not GPU supported version");
                avf = Vec.getBasicVectorFactory();
            }
        } // following Factory for CPU-Reconstruction
        else {
            avf = Vec.getBasicVectorFactory();
        }
        
        if (arg.length < 2) {
            System.out.println("Start with: config-file.xml [488] [568] [647] ...");
            return;
        }
        try {
            Conf cfg = Conf.loadFile(arg[0]);
            LiveControlPanel lcp = new LiveControlPanel(
                    cfg.r().cd("vigor-settings"), avf,
                    Arrays.copyOfRange(arg, 1, arg.length));
        } catch (Exception e) {
            System.err.println("Error loading config or initializing");
            e.printStackTrace();
            System.exit(-1);
        }
    }

}
