function [t,r,d] = segmentRateDelay(segmentEventInfo,interval)

ctb = min([segmentEventInfo.eventTime]);
ctbi = 1;
numbytes = 0;
delays = [];
N = numel(segmentEventInfo);
time = zeros(1,N);
throughput = zeros(1,N);
delay = {zeros(1,N), zeros(1,N), zeros(1,N)};
for i=1:numel(segmentEventInfo)
    ctc = segmentEventInfo(i).eventTime;
    while (ctc-ctb) > interval
        numbytes = numbytes - segmentEventInfo(ctbi).packetLength;
        delays = delays(2:end);
        ctbi = ctbi + 1;
        ctb = segmentEventInfo(ctbi).eventTime;
    end
    numbytes = numbytes + segmentEventInfo(i).packetLength;
    delays = [delays segmentEventInfo(i).currentDelay];
    time(i) = ctc;
    throughput(i) = 8*numbytes/(interval/1e9);
    delay{1}(i) = mean(delays);
    delay{2}(i) = min(delays);
    delay{3}(i) = max(delays);
end

t=time;
r=throughput;
d=delay;

end

