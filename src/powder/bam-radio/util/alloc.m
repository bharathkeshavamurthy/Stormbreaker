% ALLOC.M
%
% debug computations for potential bandwidth allocations
%
% probably not very useful. still deserves to be here.
%
% Copyright (c) 2018 Dennis Ogbe

% (1) constants/variables to study

% environment
rf_bandwidth = 5e6;
sample_rate = 46.08e6;
guard_band = 80e3; % fixme needs to be 80
num_nodes = 10;
fir_trans_bw = 60e3;
fir_atten_db = 40;

% control channel
cc_bw = 480e3;
cc_offset = 380e3;

% channelizer
ntaps = 961;
fft_mult = 16;

% we study this variable
os_to_study = 320;

% (2) constraint 1: does everything fit?

bw_avail = rf_bandwidth - 2 * (cc_offset + cc_bw/2 + guard_band);
bw_needed = num_nodes * sample_rate/os_to_study + (num_nodes - 1) * guard_band;
if bw_avail < bw_needed
    error(['bandwidth constraint violated: bw_needed: ' num2str(bw_needed) ' bw_avail: ' num2str(bw_avail)]);
end

% (3) constraint 2: does this work for channelizer params?
% TODO: This constraint might not be necessary.

all_os = [os_to_study, 160, 96, 64, 48, 40, 32, 24, 16, 8, 4];
L = fft_mult * (ntaps - 1) - ntaps + 1;
for ii=1:length(all_os)
    if mod(L, all_os(ii)) ~= 0
        error(['channelizer constraint violated: os ' num2str(all_os(ii)) ' % L = ' num2str(mod(L, all_os(ii)))]);
    end
end

% (4) compute equally-spaced allocation
bw = sample_rate/os_to_study;
space_between = (bw_avail - num_nodes * bw) / (num_nodes - 1);
spacing = bw + space_between;
first_channel = -bw_avail/2 + bw/2;
calloc = zeros(num_nodes, 1);
for ii=0:num_nodes-1
    calloc(ii+1) = first_channel + ii * spacing;
end

% (5) plot allocation + filters
cchannels = [];
channels = [];
filters = [];
fir_stop_ampl = 10^(-fir_atten_db/10);

% control channels
ccc = -rf_bandwidth/2 + cc_offset;
cc_cfreq = [ccc; -ccc];
for ii=1:length(cc_cfreq)
    lower = cc_cfreq(ii) - cc_bw/2;
    upper = cc_cfreq(ii) + cc_bw/2;
    c = struct();
    c.x = [lower-0.1, lower, upper, upper+0.1];
    c.y = [0, 0.95, 0.95, 0];
    cchannels = [cchannels, c];
end

% data channels
for ii=1:length(calloc)
    lower = calloc(ii) - bw/2;
    upper = calloc(ii) + bw/2;
    c = struct();
    c.x = [lower-0.1, lower, upper, upper+0.1];
    c.y = [0, 0.8, 0.8, 0];
    channels = [channels, c];
    filtmask = struct();
    filtmask.x = [lower - fir_trans_bw, lower, upper, upper + fir_trans_bw];
    filtmask.y = [fir_stop_ampl, 1, 1, fir_stop_ampl];
    filters = [filters filtmask];
end

figure(1)
for ii=1:length(cchannels)
    area(cchannels(ii).x, cchannels(ii).y, 'LineWidth', 0.01, 'FaceColor', 'b');
    hold on;
end
for ii=1:length(channels)
    area(channels(ii).x, channels(ii).y, 'LineWidth', 0.01)
end
for ii=1:length(filters)
    plot(filters(ii).x, filters(ii).y, 'k--', 'LineWidth', 2);
end
plot([-rf_bandwidth/2-1, -rf_bandwidth/2, rf_bandwidth/2, rf_bandwidth/2+1], [0, 1.1, 1.1, 0], 'r--', 'LineWidth', 2.5)
ylim([0, 1.15]);
xlim([-rf_bandwidth/2 - 100e3, rf_bandwidth/2 + 100e3]);
xticks(sort([cc_cfreq; calloc; 0]));
xlabel('Frequency Offset [Hz]')
hold off;