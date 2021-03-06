% ALLOC2.M
%
% visualize a discretization of the spectrum
%
% Copyright (c) 2018 Dennis Ogbe

% (1) constants
num_nodes = 10;
sample_rate = 46.08e6;
guard_band = 80e3; % fixme needs to be 80

% filters
fir_trans_bw = 60e3;
fir_atten_db = 40;

% control channel
cc_bw = 480e3;
cc_offset = 380e3;

% (2) the table mapping environment bandwidths to the minimum oversampling
% rate (maximum bandwidth) -- play with this
envs = [5e6, 8e6, 10e6, 20e6, 25e6, 40e6];
min_os = [320, 160, 64, 64, 48, 48];

assert(length(envs) == 6);
assert(length(envs) == length(min_os));

% (3) constraint: does everything fit for the maximum bandwidth?
bw_avail = zeros(1, length(envs));
for ii=1:length(envs)
    bw_avail(ii) = envs(ii) - 2 * (cc_offset + cc_bw/2 + guard_band);
    bw_needed = num_nodes * sample_rate/min_os(ii) + (num_nodes - 1) * guard_band;
    if bw_avail(ii) < bw_needed
        error(['bandwidth constraint violated: bw_needed: ' num2str(bw_needed) ' bw_avail: ' num2str(bw_avail)]);
    end
end

% (4) compute the number of possible channels and their center frequencies
nchan = zeros(1, length(envs));
cfreq = cell(length(envs), 1);
for ii=1:length(envs)
    % number of channels
    max_bw = sample_rate/min_os(ii);
    nchan(ii) = floor((bw_avail(ii)-guard_band)/(max_bw + guard_band));
    % equal channel spacing allocation
    dist = bw_avail(ii)/nchan(ii);
    start = -envs(ii)/2 + cc_offset + cc_bw/2 + guard_band + dist/2;
    cf = [start zeros(1, nchan(ii)-1)];
    for jj=2:nchan(ii)
        cf(jj) = cf(jj-1) + dist;
    end
    cfreq{ii} = round(cf);
end

% hard code 288 allocation for first env
cfreq{1} = [-1.656e6, -1.288e6, -0.920e6, -0.552e6, -0.184e6,  0.184e6,  0.552e6,  0.920e6,  1.288e6,  1.656e6];
min_os(1) = 160;
nchan(1) = length(cfreq{1});

% (5) plot for all possible scenarios -- NB the control channel is light blue
figure(1);
for ii=1:length(envs)
    subplot(3, 2, ii);

    rf_bandwidth = envs(ii);
    cchannels = [];
    channels = [];
    filters = [];
    fir_stop_ampl = 10^(-fir_atten_db/10);

    % control channels
    ccc = -rf_bandwidth/2 + cc_offset;
    cc_cfreq = [ccc; -ccc];
    for jj=1:length(cc_cfreq)
        lower = cc_cfreq(jj) - cc_bw/2;
        upper = cc_cfreq(jj) + cc_bw/2;
        c = struct();
        c.x = [lower-0.1, lower, upper, upper+0.1];
        c.y = [0, 0.95, 0.95, 0];
        cchannels = [cchannels, c];
    end

    % data channels
    bw = sample_rate/min_os(ii);
    for jj=1:length(cfreq{ii})
        lower = cfreq{ii}(jj) - bw/2;
        upper = cfreq{ii}(jj) + bw/2;
        c = struct();
        c.x = [lower-0.1, lower, upper, upper+0.1];
        c.y = [0, 0.8, 0.8, 0];
        channels = [channels, c];
        filtmask = struct();
        filtmask.x = [lower - fir_trans_bw, lower, upper, upper + fir_trans_bw];
        filtmask.y = [fir_stop_ampl, 1, 1, fir_stop_ampl];
        filters = [filters filtmask];
    end

    % plot this environment
    for jj=1:length(cchannels)
        area(cchannels(jj).x, cchannels(jj).y, 'LineWidth', 0.01, 'FaceColor', 'b');
        hold on;
    end
    for jj=1:length(channels)
        area(channels(jj).x, channels(jj).y, 'LineWidth', 0.01)
    end
    for jj=1:length(filters)
        plot(filters(jj).x, filters(jj).y, 'k--', 'LineWidth', 2);
    end
    plot([-rf_bandwidth/2-1, -rf_bandwidth/2, rf_bandwidth/2, rf_bandwidth/2+1], [0, 1.1, 1.1, 0], 'r--', 'LineWidth', 2.5)
    ylim([0, 1.15]);
    xlim([-rf_bandwidth/2 - 100e3, rf_bandwidth/2 + 100e3]);
    xticks(sort([cc_cfreq; cfreq{ii}(:)]));
    lbl = cellfun(@num2str, num2cell(0:(length(cfreq{ii})-1))', 'UniformOutput', false);
    xticklabels([' '; lbl; ' ']);
    title(['(' num2str(ii), ') ', ...
           num2str(envs(ii)/1e6) ' MHz environment, ' ...
           num2str(nchan(ii)) ' possible channels @ ' ...
           num2str(sample_rate/min_os(ii)/1e6) ' MHz max bandwidth' ]);
    hold off;
end


% (6) print C++ code to generate the table
% these are all oversampling factors we theoretically support
possible_os = [320, 160, 96, 64, 48, 40, 32, 24, 16, 8, 4];

fprintf('// This file is auto-generated by util/alloc2.m. DO NOT EDIT.\n');
fprintf('Channelization const &Channelization::get(int64_t rf_bandwidth) {\n');
fprintf('  static const std::map<int64_t,Channelization> m{\n');
for ii=1:length(envs)
    fprintf('    ');
    max_idx = find(possible_os == min_os(ii)) - 1;
    center_offsets = cfreq{ii}(:);
    fprintf('{%d, ', envs(ii));
    co = sprintf('%d, ', center_offsets);
    co = co(1:end-2);
    fprintf('{%d, {%s}}}', max_idx, co);
    if ii ~= length(envs)
        fprintf(',');
    end
    fprintf('\n');
end
fprintf('  };\n');
fprintf('  return m.at(rf_bandwidth);\n}\n');
fclose(fileID);
