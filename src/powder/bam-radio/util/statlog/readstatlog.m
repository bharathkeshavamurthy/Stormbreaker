function db = readstatlog(filename)

fd = fopen(filename);

preallocs = 1000000;

dbRoutingIndex = 1;
dbDetectedFrameIndex = 1;
dbTxSegmentIndex = 1;
dbRxBlockIndex = 1;
dbRxSegmentIndex = 1;
dbSegmentDelayIndex = 1;
dbCoDelStateIndex = 1;
dbNewFlowIndex = 1;

statline = fgetl(fd);
linenum = 1;
reverseStr = '';
while ischar(statline)
    ei = jsondecode(statline);
    
    if strcmp(ei.event, 'Routing')
        if dbRoutingIndex == 1
            dbRouting(preallocs) = ei;
            dbRouting(preallocs).eventTime = -1;
        end
        dbRouting(dbRoutingIndex) = ei;
        dbRoutingIndex = dbRoutingIndex + 1;
    elseif strcmp(ei.event, 'DetectedFrame')
        if dbDetectedFrameIndex == 1
            dbDetectedFrame(preallocs) = ei;
            dbDetectedFrame(preallocs).eventTime = -1;
        end
        dbDetectedFrame(dbDetectedFrameIndex) = ei;
        dbDetectedFrameIndex = dbDetectedFrameIndex + 1;
    elseif strcmp(ei.event, 'InvalidFrameHeader')
    elseif strcmp(ei.event, 'SentSegment')
        if dbTxSegmentIndex == 1
            dbTxSegment(preallocs) = ei;
            dbTxSegment(preallocs).eventTime = -1;
        end
        dbTxSegment(dbTxSegmentIndex) = ei;
        dbTxSegmentIndex = dbTxSegmentIndex + 1;
    elseif strcmp(ei.event, 'SentFrame')
    elseif strcmp(ei.event, 'ReceivedBlock')
        if dbRxBlockIndex == 1
            dbRxBlock(preallocs) = ei;
            dbRxBlock(preallocs).eventTime = -1;
        end
        dbRxBlock(dbRxBlockIndex) = ei;
        dbRxBlockIndex = dbRxBlockIndex + 1;
    elseif strcmp(ei.event, 'logOpened')
    elseif strcmp(ei.event, 'ReceivedCompletedSegment')
        if dbRxSegmentIndex == 1
            dbRxSegment(preallocs) = ei;
            dbRxSegment(preallocs).eventTime = -1;
        end
        dbRxSegment(dbRxSegmentIndex) = ei;
        dbRxSegmentIndex = dbRxSegmentIndex + 1;
    elseif strcmp(ei.event, 'BurstAck')
    elseif strcmp(ei.event, 'BurstSend')
    elseif strcmp(ei.event, 'SegmentDelay')
        if dbSegmentDelayIndex == 1
            dbSegmentDelay(preallocs) = ei;
            dbSegmentDelay(preallocs).eventTime = -1;
        end
        dbSegmentDelay(dbSegmentDelayIndex) = ei;
        dbSegmentDelayIndex = dbSegmentDelayIndex + 1;
    elseif strcmp(ei.event, 'CoDelState')
        if dbCoDelStateIndex == 1
            dbCoDelState(preallocs) = ei;
            dbCoDelState(preallocs).eventTime = -1;
        end
        dbCoDelState(dbCoDelStateIndex) = ei;
        dbCoDelStateIndex = dbCoDelStateIndex + 1;
    elseif strcmp(ei.event, 'NewFlow')
        if dbNewFlowIndex == 1
            dbNewFlow(preallocs) = ei;
            dbNewFlow(preallocs).eventTime = -1;
        end
        dbNewFlow(dbNewFlowIndex) = ei;
        dbNewFlowIndex = dbNewFlowIndex + 1;
    else
        disp(ei.event)
        assert(false);
    end

    if mod(linenum,1000) == 0
        msg = sprintf('line %dK', linenum/1000);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
        if max([dbRoutingIndex dbRxSegmentIndex dbRxBlockIndex dbTxSegmentIndex dbDetectedFrameIndex]) > preallocs
            assert(false)
        end
    end
    
    statline = fgetl(fd);
    linenum = linenum + 1;
end

if exist('dbRouting')
    db.routing = dbRouting(1:(dbRoutingIndex-1));
end
if exist('dbDetectedFrame')
    db.detection = dbDetectedFrame(1:(dbDetectedFrameIndex-1));
end
if exist('dbRxBlock')
    db.rxBlock = dbRxBlock(1:(dbRxBlockIndex-1));
end
if exist('dbRxSegment')
    db.rxSegment = dbRxSegment(1:(dbRxSegmentIndex-1));
end
if exist('dbCoDelState')
    db.codelState = dbCoDelState(1:(dbCoDelStateIndex-1));
end
if exist('dbNewFlow')
    db.newFlow = dbNewFlow(1:(dbNewFlowIndex-1));
end
if exist('dbSegmentDelay')
    db.segmentDelay = dbSegmentDelay(1:(dbSegmentDelayIndex-1));
end
if exist('dbTxSegment')
    db.txSegment = dbTxSegment(1:(dbTxSegmentIndex-1));
end

end
