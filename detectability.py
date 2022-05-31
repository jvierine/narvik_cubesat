#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import debris as d

def cubesat():

    # antenna gain in dB
    gain_dB=43.0
    gain_tx = 10**(gain_dB/10.0)
    gain_rx = 10**(gain_dB/10.0)

    # radar frequency
    freq=77e9
    wavel=3e8/freq

    # estimate effective area of the antenna (m^2)
    A_eff = gain_tx*(wavel**2.0)/4.0/n.pi
    print("approximate antenna diameter %1.2f m"%(n.sqrt(A_eff)))
    
    # transmit power (watts)
    power_tx=1.0
    # distances to test
    range_rx = n.linspace(1,1e3,num=100)

    # object diameters to test
    diameters_m = [1e-2,0.5e-2,2e-3,1e-3]

    # transmit pulse length (s)
    transmit_pulse_length=2e-3
    bandwidth=1.0/transmit_pulse_length

    # system noise temperature in Kelvin
    rx_temp=150.0

    for diameter_m in diameters_m:
        enr=d.hard_target_enr(gain_tx,gain_rx,wavel,power_tx,range_rx,range_rx,diameter_m,bandwidth,rx_temp)
        plt.loglog(range_rx,enr,label="d=%g m"%(diameter_m))
    plt.legend()
    plt.axhline(20,label="Detection limit")
    plt.xlabel("Range (m)")
    plt.ylabel("SNR")
    plt.show()

cubesat()

    
    
    

