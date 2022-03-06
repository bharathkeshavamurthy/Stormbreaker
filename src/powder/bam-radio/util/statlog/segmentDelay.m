function [t,d] = segmentDelay(sd,srcip,dstip,proto,srcport,dstport,interval)

ctb = min([sd.eventTime]);
ctbi = 1;
t = [];
d = [];
for i=1:numel(sd)
    if ~strcmp(sd(i).srcAddr,srcip) || ~strcmp(sd(i).dstAddr,dstip) || sd(i).protocol ~= proto || sd(i).srcPort ~= srcport || sd(i).dstPort ~= dstport
        continue;
    end
    ctc = sd(i).eventTime;
    while (ctc-ctb) > interval
        ctbi = ctbi + 1;
        ctb = sd(ctbi).eventTime;
    end
    t = [t ctc];
    d = [d sd(i).delay];
end

end

