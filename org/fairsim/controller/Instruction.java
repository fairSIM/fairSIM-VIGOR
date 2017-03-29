/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fairsim.controller;

import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Class that defines the Objekts for the communikation between the GUI and Client <br>
 * SLM - Server - Client - GUI
 * @author m.lachetta
 */
public class Instruction {
    
    Lock lock;
    Condition condition;
    String command;
    
    Instruction(String command) {
        lock = new ReentrantLock();
        condition = lock.newCondition();
        this.command = command;
    }
}
