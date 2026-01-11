from deltasigma import *
from matplotlib import figure
from matplotlib.pyplot import *
from matplotlib.table import Table
from itertools import zip_longest as izip_longest
import numpy as np


order = 5
osr = 32
nlev = 2
f0 = 0.
Hinf = 1.5
form = 'CRFB'

ntf = synthesizeNTF(order, osr, 2, Hinf, f0)            # Optimized zero placement
print("Synthesized a %d-order NTF, with roots:\n" % order)
print(" Zeros:\t\t\t Poles:")
for z, p in zip(ntf[0], ntf[1]):
    print("(%f, %fj)\t(%f, %fj)" % (np.real(z), np.imag(z), np.real(p), np.imag(p)))
print("")
print("The NTF transfer function has the following expression:\n")
print(pretty_lti(ntf))
print("")



plotPZ(ntf, showlist=True)



a, g, b, c = realizeNTF(ntf, form)
b = np.hstack(( # Use a single feed-in for the input
               np.atleast_1d(b[0]),
               np.zeros((b.shape[0] - 1, ))
             ))
ABCD = stuffABCD(a, g, b, c, form)
print("ABCD Matrix:")
print(ABCD)

figure(figsize=(15,8))
PlotExampleSpectrum(ntf, M=1, osr=osr, f0=f0)



snr, amp = simulateSNR(ntf, osr, None, f0, nlev)



figure(figsize=(15,8))
if nlev == 2:
    snr_pred, amp_pred, k0, k1, se = predictSNR(ntf, osr)
    plot(amp_pred, snr_pred, '-', label='predicted')
plot(amp, snr,'o-.g', label='simulated')
xlabel('Input Level (dBFS)')
ylabel('SQNR (dB)')
peak_snr, peak_amp = peakSNR(snr, amp)
msg = 'peak SQNR = %4.1fdB  \n@ amp = %4.1fdB  ' % (peak_snr, peak_amp)
text(peak_amp-10,peak_snr,msg, horizontalalignment='right', verticalalignment='center');
msg = 'OSR = %d ' % osr
text(-2, 5, msg, horizontalalignment='right');
figureMagic([-100, 0], 10, None, [0, 100], 10, None, [12, 6], 'Time-Domain Simulations')
legend(loc=2);



print('Doing dynamic range scaling... ')
ABCD0 = ABCD.copy()
ABCD, umax, S = scaleABCD(ABCD0, nlev, f0)
print('Done.')
print("Maximum input magnitude: %.3f" % umax)


print('Verifying dynamic range scaling... ')
u = np.linspace(0, 0.95*umax, 30)
N = 1e4
N0 = 50
test_tone = np.cos(2*np.pi*f0*np.arange(N))
test_tone[:N0] = test_tone[:N0]*(0.5 - 0.5*np.cos(2*np.pi/N0*np.arange(N0)))
maxima = np.zeros((order, u.shape[0]))
for i in np.arange(u.shape[0]):
    ui = u[i]
    v, xn, xmax, y = simulateDSM(ui*test_tone, ABCD, nlev)
    maxima[:, i] = xmax[:, 0]
    if (xmax > 1e2).any(): 
        print('Warning, umax from scaleABCD was too high.')
        umax = ui
        u = u[:i]
        maxima = maxima[:, :i]
        break
print('Done.')
print("Maximum DC input level: %.3f" % umax)




colors = get_cmap('jet')(np.linspace(0, 1.0, order))
for i in range(order):
    plot(u,maxima[i,:], 'o-', color=colors[i], label='State %d' % (i+1))
grid(True)
#text(umax/2, 0.05, 'DC input', horizontalalignment='center', verticalalignment='center')
figureMagic([0, umax], None, None, [0, 1] , 0.1, 2, [12, 6], 'State Maxima')
xlabel('DC input')
ylabel('Maxima')
legend(loc='best');




a, g, b, c = mapABCD(ABCD, form)



adc = {
       'order':order,
       'osr':osr,
       'nlev':nlev,
       'f0':f0,
       'ntf':ntf,
       'ABCD':ABCD,
       'umax':umax,
       'peak_snr':peak_snr,
       'form':form,
       'coefficients':{
                       'a':a,
                       'g':g,
                       'b':b,
                       'c':c
                      }
      }





ax = gca()
ilabels = ['#1', '#2', '#3', '#4', '#5', '#6']
rows = []
rows.append(['Coefficients', 'DAC feedback', 'Resonator feedback', 'Feed-in', 'Interstage'])
rows.append(['', 'a(n)', 'g(n)', 'b(n)', 'c(n)'])
for x in izip_longest(ilabels,
                      adc['coefficients']['a'].tolist(),
                      adc['coefficients']['g'].tolist(),
                      adc['coefficients']['b'].tolist(),
                      adc['coefficients']['c'].tolist(), fillvalue=""):
    rows.append(list(x))
ax.axis('off')
table = ax.table(cellText=rows, loc='center')
table.auto_set_font_size(False)
table.scale(1, 1.5)

