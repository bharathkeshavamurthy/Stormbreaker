srnid = [118 119 120 122 123];
RESNUM = 31372;
throughputInt = 1e9;
blerInt = 1e9;
stateInt = 1e3;
hexstr = '607197c29f1f';
%dirfstr = ['~/RESERVATION-%d/bamwireless-codel-' hexstr '-srn%%d-RES%d/stat.log'];
%srnfstr = sprintf(dirfstr, RESNUM, RESNUM);
%srnfstr = '/sgl/bam-radio/sgl.test/radio%d/stat.log';
srnfstr = '~/stat-%d.log';

if ~exist('jstats')
    parfor i=1:length(srnid)
        jstats{i} = readstatlog(sprintf(srnfstr, srnid(i)));
    end
end


fignum=1;
close(figure(fignum));figure(fignum);fignum=fignum+1;

ii=[];
parfor i=1:length(srnid)
    try
        [t{i},r{i},d{i}]=segmentRateDelay(jstats{i}.rxSegment,throughputInt);
        ii = [ii i];
    catch
    end
end
ts = zeros(1,numel(ii)) + inf;
for i=ii
    ts(i) = min(t{i});
end

lgsrnid = cell(1,numel(ii));
tp = cell(1,max(ii));
hrx = cell(1,3);
for i = 1:3
    hrx{i} = gobjects(1,numel(ii));
end
j = 0;
for i=ii
    j = j + 1;
    tp{i}=(t{i}-min(ts))/1e9;
    subplot(5,1,1);
    hold on;
    plot(tp{i}, r{i}/1e6)
    title('RX Throughput')
    xlabel('time (s)')
    ylabel('rate (Mbps)')
    subplot(5,1,2);
    hold on;
    hrx{1}(j) = plot(tp{i}, d{i}{1}/1e6);
    hrx{2}(j) = plot(tp{i}, d{i}{2}/1e6, '--', 'Color', hrx{1}(j).Color);
    hrx{3}(j) = plot(tp{i}, d{i}{3}/1e6, '--', 'Color', hrx{1}(j).Color);
    ylim([0 10000])
    title('RX Delay (avg) (TUN->DLLDEF)')
    xlabel('time (s)')
    ylabel('delay (ms)')
    lgsrnid{j} = sprintf('SRN %d', srnid(i));
end

legend(hrx{1}, lgsrnid);

ii=[];
parfor i=1:length(srnid)
    try
        [t{i},bler{i}]=blockErrorRate(jstats{i}.rxBlock,srnid,blerInt);
        ii = [ii i];
    catch
    end
end
ts = zeros(1,numel(ii)) + inf;
for i=ii
    ts(i) = min(t{i});
end

lgsrnid = cell(1,numel(ii));
j = 0;
for i=ii
    j = j + 1;
    tp{i}=(t{i}-min(ts))/1e9;
    subplot(5,1,3);
    hold on;
    plot(tp{i}, log10(bler{i}));
    title('RX Block Error Rate (avg)')
    xlabel('time (s)')
    ylabel('BLER')
    lgsrnid{j} = sprintf('SRN %d', srnid(i));
end
legend(lgsrnid);

ii=[];
parfor i=1:length(srnid)
    try
        [t{i},r{i},d{i}]=segmentRateDelay(jstats{i}.txSegment,throughputInt);
        ii = [ii i];
    catch
    end
end
ts = zeros(1,numel(ii)) + inf;
for i=ii
    ts(i) = min(t{i});
end

lgsrnid = cell(1,numel(ii));
htx = cell(1,3);
for i = 1:3
    htx{i} = gobjects(1,numel(ii));
end
j = 0;
for i=ii
    j = j + 1;
    tp{i}=(t{i}-min(ts))/1e9;
    subplot(5,1,4);
    hold on;
    plot(tp{i}, r{i}/1e6)
    title('TX Throughput')
    xlabel('time (s)')
    ylabel('rate (Mbps)')
    subplot(5,1,5);
    hold on;
    htx{1}(j) = plot(tp{i}, d{i}{1}/1e6);
    htx{2}(j) = plot(tp{i}, d{i}{2}/1e6, '--', 'Color', htx{1}(j).Color);
    htx{3}(j) = plot(tp{i}, d{i}{3}/1e6, '--', 'Color', htx{1}(j).Color);
    ylim([0 10000])
    title('TX Delay (avg) (TUN->MOAB)')
    xlabel('time (s)')
    ylabel('delay (ms)')
    lgsrnid{j} = sprintf('SRN %d', srnid(i));
end
legend(htx{1}, lgsrnid);

for j = 1:numel(ii)
uicontrol('Style', 'checkbox', 'String', lgsrnid{j}, ...
    'Value', 1, 'Position', [10,10+20*(numel(ii)-j),80,20], ...
    'Callback', {@toggle_line, j, hrx, htx});
end

ii=[];
parfor i=1:length(srnid)
    try
        [t{i},lavg{i},dropping{i},qs{i}]=codelState(jstats{i}.codelState,stateInt);
        ii = [ii i];
    catch
    end
end
ts = zeros(1,numel(ii)) + inf;
for i=ii
    ts(i) = min(t{i});
end

lgsrnid = cell(1,numel(ii));
j = 0;
for i=ii
    close(figure(fignum));figure(fignum);fignum=fignum+1;
    j = j + 1;
    tp{i}=(t{i}-min(ts))/1e9;
    subplot(2,1,1);
    hold on;
    plot(tp{i}, lavg{i}/1e6);
    title([sprintf('SRN %d', srnid(i)) 'CoDel Average Latency'])
    xlabel('time (s)')
    ylabel('Latency (ms)')
    subplot(2,1,2);
    hold on;
    plot(tp{i}, dropping{i});
    title([sprintf('SRN %d', srnid(i)) ' dropping'])
    xlabel('time (s)')
    ylabel('dropping')
end

function toggle_line(hObj, ~, j, hrx, htx)
    if hObj.Value
        state = 'on';
    else
        state = 'off';
    end
    for i = 1:3
        hrx{i}(j).Visible = state;
        htx{i}(j).Visible = state;
    end
end
