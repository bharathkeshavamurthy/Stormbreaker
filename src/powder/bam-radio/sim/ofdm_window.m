%% quick sanity simulation for windowed DFT-s-OFDM

%% prereqs
addpath('../util');
% make sure the debug proto defs are loaded
if length(dir('debug_pb2.py')) == 0
    system(['LD_LIBRARY_PATH="" protoc -I=../controller/src --python_out=. ../controller/src/debug.proto'])
end

% some real-world frame parameters (20x oversampling, etc)
params = load_frame_params('params.bin');
nfft = params.symbols(1).symbol_length;       % 128
os = params.symbols(1).oversample_rate;       % 20

%ncp = params.symbols(1).cyclic_prefix_length; % 12
ncp = 12;
nw = 6; % window size

% we want to compare sidelobes for
% (a) regular cyclic prefix
% (b) windowed cyclic prefix + overlapped postfix (same overall length)
% to do this, we first modulate the frame without windowing

% random QPSK symbols
qpsk = @(N) randsample(1/sqrt(2) * [1 -1], N, true).' ...
       + 1j * randsample(1/sqrt(2) * [1 -1], N, true).';

% carrier mapping (random data, real pilots)
data1 = zeros(params.symbols(1).symbol_length, params.num_symbols);
for ii=1:params.num_symbols
    sym = params.symbols(ii);
    data1(sym.pilot_carrier_mapping, ii) = sym.pilot_symbols;
    if sym.num_data_carriers > 0
        d = qpsk(sym.num_data_carriers);
        ds = fft(d, params.dft_spread_length);
        data1(sym.data_carrier_mapping, ii) = ds;
    end
end

% fft + oversampling
data2 = zeros(os*nfft, params.num_symbols);
data2(1:nfft/2, :) = data1(1:nfft/2, :);
data2(end-nfft/2+1:end, :) = data1(nfft/2+1:end, :);
td1 = ifft(data2); % MATLAB IFFT != FFTW (scale factor)

% cyclic prefix (regular)
rcp_td2 = [td1(end-os*ncp+1:end, :); td1];

% cyclic prefix (windowed + postfix)

% construct and plot the windows for prefix and postfix
% ref: http://liquidsdr.org/doc/ofdmflexframe/ (there is a type in the def
% of the window function...)
wn = @(n) (sin(pi*((n(:))+0.5)/(nw*os*2))).^2;
head = wn((0:(nw*os-1)));
tail = flipud(head);
figure(2);
plot(head, 'k');
hold on;
plot(tail, 'r');
hold off;
legend('head', 'tail');
title('sin^2 window')

wcp_td2 = [td1(end-os*ncp+1:end, :); td1; td1(1:os*nw, :)];
window = [repmat(head, 1, params.num_symbols); ...
          ones(size(wcp_td2, 1)-os*nw*2, params.num_symbols); ...
          repmat(tail, 1, params.num_symbols)];
wcp_td2 = wcp_td2 .* window;
wcp_td3 = zeros(params.num_symbols*os*(ncp+nfft)+os*nw, 1);
wcp_td3(1:size(wcp_td2, 1)) = wcp_td2(:, 1);
for ii=2:params.num_symbols
    sym = wcp_td2(:, ii);
    add_start = (ii-1) * (os*(ncp+nfft)) + 1;
    add_end = add_start + os*nw - 1;
    wcp_td3(add_start:add_end) = wcp_td3(add_start:add_end) + sym(1:os*nw);
    wcp_td3(add_end+1:(add_end+os*(ncp+nfft))) = sym(os*nw+1:end);
end

%% plot
Fs = 20e6;
nfft2 = 2^nextpow2(length(rcp_td2(:)));
f = Fs/(nfft2) * [-nfft2/2:nfft2/2-1];

fd_regular_cp = 10 * log10(abs(fftshift(fft(rcp_td2(:), nfft2))));
fd_window_cp = 10 * log10(abs(fftshift(fft(wcp_td3, nfft2))));

figure(1);
plot(f, fd_regular_cp);
hold on;
plot(f, fd_window_cp, 'r');
xlabel('Frequency [Hz]');
ylabel('Amplitude [dB]');
% xlim(0.8 * [-1 1] * 1e6);
% ylim([-20, 30])
legend('no window', ['window N_{w} = ' num2str(nw)]);
title(['N_{cp} = ' num2str(ncp)]);
hold off;