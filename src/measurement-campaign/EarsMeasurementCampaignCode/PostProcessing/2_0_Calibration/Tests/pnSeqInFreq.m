% PNSEQINFREQ Illustrate how a PN sequence will look like in frequence
% domain.
%
% Yaguang Zhang, Purdue, 09/08/2017

% Get a PN sequence.
rng default;
pnSeq = randi([0,1],2096,1);

% Assumptions.
Fs = 1000;
x = pnSeq;
t = 0:1/Fs:1-1/Fs;

% FFT.
N = length(x);
xdft = fft(x);
xdft = xdft(1:N/2+1);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:Fs/length(x):Fs/2;

plot(freq,10*log10(psdx))
grid on
title('Periodogram Using FFT')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')

% EOF