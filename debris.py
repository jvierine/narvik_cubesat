#!/usr/bin/env python

import numpy as n
import matplotlib.pyplot as plt
import scipy.constants as c
import scipy.stats as stats
import scipy.special as special
import scipy.signal as ss
import scipy.interpolate as si

# bandwidth is a compound bandwidth. it is the bandwidth of the noise after coherent integration.
# if, say, the original bandwidth is 1e6, and you integrate four 1000 sample transmit pulses, the
# bandwidth is 1000 Hz
def hard_target_enr(gain_tx, gain_rx, wavelength_m, power_tx, range_tx_m, range_rx_m, diameter_m=0.01, bandwidth=10, rx_noise_temp=150.0):
    ##
    ## Determine returned signal power, given diameter of sphere
    ## Ignore Mie regime and use either optical or rayleigh scatter
    ##
    is_rayleigh = diameter_m < wavelength_m/(n.pi*n.sqrt(3.0)) 
    is_optical = diameter_m >= wavelength_m/(n.pi*n.sqrt(3.0))
    rayleigh_power = (9.0*power_tx*(((gain_tx*gain_rx)*(n.pi**2.0)*(diameter_m**6.0))/(256.0*(wavelength_m**2.0)*(range_rx_m**2.0*range_tx_m**2.0))))
    optical_power = (power_tx*(((gain_tx*gain_rx)*(wavelength_m**2.0)*(diameter_m**2.0)))/(256.0*(n.pi**2)*(range_rx_m**2.0*range_tx_m**2.0)))
    rx_noise = c.k*rx_noise_temp*bandwidth
    return(((is_rayleigh)*rayleigh_power + (is_optical)*optical_power)/rx_noise)

def target_diameter(gain_tx, gain_rx, wavelength_m, power_tx, range_tx_m, range_rx_m, enr=1.0, bandwidth=10.0, duty_cycle=0.2, rx_noise_temp=150.0):
    ##
    ## determine smallest sphere detactable with a certain enr
    ## Ignore Mie regime and use either optical or rayleigh scatter
    ##
    n_diam=1000
    diams = 10**n.linspace(-4,3,num=n_diam)
    diams[n_diam-1]=n.nan
    enrs = hard_target_enr(gain_tx,gain_rx,wavelength_m,power_tx,range_tx_m,range_rx_m,diams,bandwidth,rx_noise_temp)
    i=0
    while enrs[i] < enr and i<n_diam:
        i+=1
    return(diams[i])

# simulate echo with range and doppler
def simulate_echo(codes,t_vecs,bw=1e6,dop_Hz=0.0,range_m=1e3,plot=False,sr=5e6):
    codelen=len(codes[0])
    n_codes=len(codes)
    tvec=n.zeros(codelen*n_codes)
    z=n.zeros(codelen*n_codes,dtype=n.complex64)
    for ci,code in enumerate(codes):
        z[n.arange(codelen)+ci*codelen]=code
        tvec[n.arange(codelen)+ci*codelen]=t_vecs[ci]
    tvec_i=n.copy(tvec)
    tvec_i[0]=tvec[0]-1e99
    tvec_i[len(tvec)-1]=tvec[len(tvec)-1]+1e99        
    zfun=si.interp1d(tvec_i,z,kind="linear")
    dt = 2.0*range_m/c.c

    z=zfun(tvec+dt)*n.exp(1j*n.pi*2.0*dop_Hz*tvec)
    if plot:
        plt.plot(tvec,z.real)
        plt.show()
    return(z)

# calculate line of sight range and range-rate error,
# given ENR after coherent integration (pulse compression)
# txlen in microseconds
def lin_error(enr=10.0,txlen=1000.0,n_ipp=10,ipp=20e-3,bw=1e6,dr=15.0,ddop=1.0,sr=50e6,plot=False):
    codes=[]
    t_vecs=[]    
    n_bits = int(bw*txlen/1e6)
    oversample=int(sr/bw)
    wfun=ss.hamming(oversample)
    wfun=wfun/n.sum(wfun)
    for i in range(n_ipp):
        bits=n.array(n.sign(n.random.randn(n_bits)),dtype=n.complex64)
        zcode=n.zeros(n_bits*oversample+2*oversample,dtype=n.complex64)
        for j in range(oversample):
            zcode[n.arange(n_bits)*oversample+j+oversample]=bits
        # filter signal so that phase transitions are not too sharp
        zcode=n.convolve(wfun,zcode,mode="same")
        codes.append(zcode)
        tcode=n.arange(n_bits*oversample+2*oversample)/sr + float(i)*ipp
        t_vecs.append(tcode)

    z0=simulate_echo(codes,t_vecs,dop_Hz=0.0,range_m=0.0,bw=bw,sr=sr)
    tau=(float(n_ipp)*txlen/1e6)
    # convert coherently integrated ENR to SNR
    snr = enr/(tau*sr)

    if plot:
        plt.plot(z0.real)
        plt.plot(z0.imag)
        plt.show()

    z_dr=simulate_echo(codes,t_vecs,dop_Hz=0.0,range_m=dr,bw=bw,sr=sr)
    if plot:
        plt.plot(z_dr.real)
        plt.plot(z_dr.imag)
        plt.show()
    z_ddop=simulate_echo(codes,t_vecs,dop_Hz=ddop,range_m=0.0,bw=bw,sr=sr)
    z_diff_r=(z0-z_dr)/dr
    z_diff_dop=(z0-z_ddop)/ddop
    if plot:
        plt.plot(z_diff_r.real)
        plt.plot(z_diff_r.imag)
        plt.show()
        plt.plot(z_diff_dop.real)
        plt.plot(z_diff_dop.imag)
        plt.show()

    t_l=len(z_dr)
    A=n.zeros([t_l,2],dtype=n.complex64)
    A[:,0]=z_diff_r
    A[:,1]=z_diff_dop
    S=n.real(n.linalg.inv(n.dot(n.transpose(n.conj(A)),A))/snr)
    

    return(n.sqrt(n.diag(S)))

#
# Simulate the effect of the planned "satellite filter".
#
def debris_filter():
    n_r=1000
    ranges = n.linspace(300e3,2000e3,num=n_r)
    filter_diams=n.zeros(n_r)
    diams=n.zeros(n_r)    
    for ri,r in enumerate(ranges):
        filter_diams[ri] = target_diameter(10**4.3, 10**1.9, 230e6/c.c, 10e6, ranges[ri], ranges[ri], enr=100.0, bandwidth=2000.0, duty_cycle=0.2, rx_noise_temp=150.0)
        diams[ri] = target_diameter(10**4.3, 10**4.3, 230e6/c.c, 10e6, ranges[ri], ranges[ri], enr=10.0, bandwidth=200.0, duty_cycle=0.2, rx_noise_temp=150.0)
    plt.semilogy(ranges/1e3,filter_diams,label="Filtered objects")
    plt.semilogy(ranges/1e3,diams,label="Detectable objects")
    plt.xlabel("Range (km)")
    plt.ylabel("Target diameter (m)")    
    plt.grid()
    plt.legend()
    plt.savefig("sat_filter.png")
    plt.show()

def tx_len_sweep(tlen=n.linspace(10.0,2000.0,num=10),bw=1e6,enr=1000.0,n_ipp=10,ipp=20e-3):
    drs=[]
    ddops=[]    
    for tl in tlen:
        dr,ddop=lin_error(enr=enr,txlen=tl,bw=1e6,n_ipp=n_ipp,ipp=ipp)
        drs.append(dr)
        ddops.append(ddop)
    plt.subplot(121)        
    plt.semilogy(tlen,drs)
    plt.title("Fixed ENR=%1.0f n_ipp=%d IPP=%1.2f (ms)"%(enr,n_ipp,ipp*1e3))
    plt.xlabel("TX pulse length ($\mu$s)")
    plt.ylabel("1-$\sigma$ Range error (m)")
    plt.grid()        
    plt.subplot(122)
    plt.semilogy(tlen,ddops)
    plt.title("Fixed ENR=%1.0f"%(enr))
    plt.xlabel("TX pulse length ($\mu$s)")
    plt.ylabel("1-$\sigma$ Doppler error (Hz)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("deb_tx_len_sweep_const_enr.png")

    plt.show()

def ipp_sweep(n_ipps=n.arange(1,20),bw=1e6,enr=1000.0,ipp=20e-3,txlen=2000.0):
    drs=[]
    ddops=[]    
    for n_ipp in n_ipps:
        print(n_ipp)
        ratio=float(n_ipp)/float(n.max(n_ipps))
        print(ratio)        
        dr,ddop=lin_error(enr=ratio*enr,txlen=txlen,bw=1e6,n_ipp=n_ipp,ipp=ipp)
        drs.append(dr)
        ddops.append(ddop)
    plt.subplot(121)        
    plt.semilogy(n_ipps,drs)
    plt.title("Maximum ENR=%1.0f IPP=%1.2f (ms)"%(enr,ipp*1e3))
    plt.xlabel("Number of IPPs to integrate")
    plt.ylabel("1$\sigma$ Range error (m)")
    plt.grid()        
    plt.subplot(122)
    plt.semilogy(n_ipps,ddops)
    plt.xlabel("Number of IPPs to integrate")
    plt.ylabel("1$\sigma$ Doppler error (Hz)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("deb_n_ipp_sweep.png")

    plt.show()

def tx_len_sweep2(tlen=n.linspace(10.0,2000.0,num=10),bw=1e6,enr0=1000.0,n_ipp=20):
    drs=[]
    ddops=[]
    maxtlen=n.max(tlen)
    for tl in tlen:
        dr,ddop=lin_error(enr=(tl/maxtlen)*enr0,txlen=tl,bw=bw,n_ipp=n_ipp)
        drs.append(dr)
        ddops.append(ddop)
    plt.subplot(121)
    plt.loglog(tlen,drs)
    plt.title("Fixed $P_{\mathrm{tx}}$, Max ENR=%1.0f"%(enr0))
    plt.xlabel("TX pulse length ($\mu$s)")
    plt.ylabel("1-$\sigma$ Range error (m)")
    plt.grid()        
    plt.subplot(122)
    plt.loglog(tlen,ddops)
    plt.title("Fixed $P_{\mathrm{tx}}$, Max ENR=%1.0f"%(enr0))
    plt.xlabel("TX pulse length ($\mu$s)")
    plt.ylabel("1-$\sigma$ Doppler error (Hz)")
    plt.grid()    
    plt.tight_layout()
    plt.savefig("deb_tx_len_sweep_const_txp.png")

    plt.show()
    
def bw_sweep(bw=n.linspace(0.01e6,5e6,num=10),txlen=1000.0,enr=100.0,n_ipp=20,ipp=20e-3):
    drs=[]
    ddops=[]    
    for b in bw:
        print("bw %1.2f"%(b))
        dr,ddop=lin_error(enr=enr,txlen=txlen,bw=b,n_ipp=n_ipp)
        drs.append(dr)
        ddops.append(ddop)
    plt.subplot(121)
    plt.loglog(bw/1e6,drs)
    plt.xlabel("TX bandwidth (MHz)")
    plt.ylabel("1-$\sigma$ Range error (m)")
    plt.title("ENR=%1.0f txlen=%1.2f (ms)\nn_ipp=%d ipp=%1.2f (ms)"%(enr,txlen/1e3,n_ipp,ipp*1e3))
    plt.grid()        
    plt.subplot(122)    
    plt.semilogy(bw/1e6,ddops)
    plt.title("Fixed ENR=%1.0f"%(enr))    
    plt.xlabel("TX bandwidth (MHz)")
    plt.ylabel("1-$\sigma$ Doppler error (Hz)")
    plt.grid()    
    plt.tight_layout()
    plt.savefig("deb_bw_sweep_lin.png")
    plt.show()

#
# Sweep through different ENRs and calcul
#
def enr_sweep(enrs=10.0**n.linspace(0,6.0,num=10),txlen=1000.0,bw=1e6,n_ipp=10,ipp=20e-3):
    drs=[]
    ddops=[]    
    for s in enrs:
        dr,ddop=lin_error(enr=s,txlen=txlen,bw=bw,n_ipp=n_ipp)
        drs.append(dr)
        ddops.append(ddop)
    plt.subplot(121)
    plt.semilogy(10.0*n.log10(enrs),drs)
    plt.title("TX len %1.0f $\mu$s IPP=%1.2f (ms)\nn_ipp=%d TX BW %1.0f MHz"%(txlen,ipp*1e3,n_ipp,bw/1e6))
    plt.xlabel("ENR (dB)")
    plt.ylabel("1-$\sigma$ Range error (m)")
    plt.grid()        
    plt.subplot(122)    
    plt.semilogy(10.0*n.log10(enrs),ddops)
    plt.xlabel("ENR (dB)")
    plt.ylabel("1-$\sigma$ Doppler error (Hz)")
    plt.grid()    
    plt.tight_layout()
    plt.savefig("deb_enr_sweep_lin.png")
    plt.show()
    
if __name__ == "__main__":
    ipp_sweep()    
    enr_sweep()
    tx_len_sweep()
    tx_len_sweep2()
    bw_sweep()    

