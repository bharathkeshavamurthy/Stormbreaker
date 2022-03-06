% VANILLADSPVSPWELCH Compare the DSP results gotten by the vanilla FFT
% approach and the comman pwelch.
%
% Yaguang Zhang, Purdue, 09/21/2017

rng default

% Input data.
fs = 1000;
t = 0:1/fs:5-1/fs;

noisevar = 1/4;
x = cos(2*pi*100*t)+sqrt(noisevar)*randn(size(t));

% PSD by pwelch.
[pxx,f] = pwelch(x,500,300,500,fs,'centered','power');

% PSD by fft.
L0 = length(x);
Y0 = fftshift(fft(x));
f0 = (-L0/2:L0/2-1)*(fs/L0);
psd0 = abs(Y0).^2/L0;

% Figure.
figure; hold on;
hFft = plot(f0,10*log10(psd0), '-r');
hPwelch = plot(f,10*log10(pxx), '-.b');
hold off;
legend([hPwelch, hFft], 'PSD by pwelch', 'PSD by fft');
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)')
grid

% EOF