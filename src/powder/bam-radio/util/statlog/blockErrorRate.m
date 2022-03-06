function [t,bler] = blockErrorRate(rxBlock,srcNodeIDfilter,interval)

ctb = min([rxBlock.eventTime]);
ctbi = 1;
bv = [];
t = [];
bler = [];
for i=1:numel(rxBlock)
    if ~ismember(rxBlock(i).srcNodeID,srcNodeIDfilter)
        continue;
    end
    ctc = rxBlock(i).eventTime;
    while (ctc-ctb) > interval
        bv = bv(2:end);
        ctbi = ctbi + 1;
        ctb = rxBlock(ctbi).eventTime;
    end
    bv = [bv 1-rxBlock(i).valid];
    t = [t ctc];
    bler = [bler mean(bv)];
end

end

