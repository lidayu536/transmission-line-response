import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def theta_sl(i, lam3, lam2, thi, thf):
    han = (1 - lam3) * (1 - np.cos(2*np.pi*i)) + lam2 * (1 - np.cos(4*np.pi*i)) + lam3 * (1 - np.cos(6*np.pi*i))
    return thi + (thf - thi) / 2.0 * han

def ACZWave(amp, duration, sampling_rate, lam2=-0.18, lam3=0.04, thf=0.864, thi=0.05):
    ts=np.arange(0, duration, 1/sampling_rate)
    nts = ts/duration
    n = len(nts)
    th_sl = [theta_sl(i, lam3, lam2, thi, thf) for i in nts]
    tlu = np.cumsum(np.sin(th_sl))
    th = np.zeros(n)
    j = 0
    for i in range(n):
        while tlu[j] < i * tlu[-1] / n and j < n:
            j += 1
        th[i] = th_sl[j]
    thi = 1 / np.tan(th) - 1 / np.tan(th[0])
    ws = amp / np.min(thi) * thi
    
    return ts, ws

def FlattopWave(amp, duration, sampling_rate, edge=4): 
    ts = np.arange(0, duration, 1.0/sampling_rate)
    waveform= np.zeros(len(ts))
    for i in range(len(ts)):
        if ts[i] < edge:
            waveform[i] = 0.5*(1-np.cos(np.pi*ts[i]/(edge)))
        elif ts[i] > (duration-edge):
            waveform[i] = 0.5*(1+np.cos(np.pi*(ts[i]-(duration-edge))/(edge)))
        else:
            waveform[i] = 1
        waveform=waveform*amp
    return ts, waveform

def CosEnv(amp, duration, sampling_rate):
    ts = np.arange(0, duration, 1.0/sampling_rate)
    waveform = amp* (1-np.cos(2*np.pi*ts/duration))
    return waveform

def Differential(waveform, sampling_rate, 
                 *, diff_multiple=10,
                 ):
    dt = 1/sampling_rate
    duration = len(waveform)*dt
    ts = np.arange(0, duration, 1.0/sampling_rate)
    spline = interp1d(ts, waveform, kind='cubic', fill_value="extrapolate")
    diff_waveform = (spline(ts+dt/diff_multiple)-spline(ts-dt/diff_multiple))/(2*dt/diff_multiple)
    return diff_waveform


def show_spectrum(waveform, sampling_rate, idle_extend=0, xlim=None, ylim=None, xlabel="Frequency(GHz)", ylabel="a.u.", is_log=True):
    dt=1/sampling_rate
    # 确保输入为numpy数组
    waveform = np.asarray(waveform)
    
    # 如果需要扩展零来减小边界效应
    if idle_extend>1e-9:
        extend_len = int(idle_extend/dt) +1
        waveform_extend = np.pad(waveform, (0, extend_len), mode='constant')
    else:
        waveform_extend = waveform
    freqs = np.fft.fftfreq(len(waveform_extend), dt)
    
    # 对波形进行FFT
    waveform_fft = np.fft.fft(waveform_extend)/len(waveform_extend)
    
    plt.figure()
    if is_log:
        plt.plot(np.fft.fftshift(freqs), 20*np.log10(abs(np.fft.fftshift(waveform_fft))))
    else:
        plt.plot(np.fft.fftshift(freqs), abs(np.fft.fftshift(waveform_fft)))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
    plt.figure()
    plt.plot(np.fft.fftshift(freqs), np.angle(np.fft.fftshift(waveform_fft)))
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel("phase(rad)")
    plt.grid(True)
    plt.show()
    return freqs, waveform_fft

    
    
def gen_alpha_drag(waveform, sampling_rate, alpha=1.0, delta=1.0):
    waveform_diff = Differential(waveform, sampling_rate)
    return waveform + 1j*alpha/delta*waveform_diff

