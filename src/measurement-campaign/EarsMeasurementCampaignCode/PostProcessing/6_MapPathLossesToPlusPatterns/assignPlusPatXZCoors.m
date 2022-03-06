function [xs, zs, hMissingSamps] ...
    = assignPlusPatXZCoors(step, numSampsPerAxis, timestamps)
%ASSIGNPLUSPATXZCOORS Deal with sample missing cases and map the samples
%back to the (x,z) plus shaped measurement pattern.
%
% Input:
%    - step
%      Step size in meter.
%    - numSampsPerAxis
%      Number of samples per axis (i.e. 2*numSampsPerAxis samples in total
%      for the "+" pattern).
%    - timestamps
%      Used to check whether there is any sample missing and if so, how to
%      properly assign the coordinates.
%
% Output:
%    - xs, zx
%      Column vectors specifying the coordinates on the plus pattern. We
%      will have a right-hand coordinate system of (x, z) in meter where
%      the RX antenna points at +y. More specificially, +x is the
%      right-hand side of the antenna and +z is the top of it. (0, 0) is
%      the home position (x at center, z at top) of the platform.
%    - hMissingSamps
%      If there are any sample missing, a plot will be generated to show
%      them.
%
% Yaguang Zhang, Purdue, 10/18/2017

% Start at home (0,0), moving:
%   -> (2*numSampsPerAxis-1) -z -> numSampsPerAxis/2 +z numSampsPerAxis/2
%   -x -> numSampsPerAxis +x
xs = [zeros((numSampsPerAxis),1); ...
    ((0:(numSampsPerAxis-1))' ...
    - ones(numSampsPerAxis,1).*(numSampsPerAxis/2)).*step];
zs = [-(0:(numSampsPerAxis-1))'.*step; ...
    -ones(numSampsPerAxis,1).*(numSampsPerAxis/2-1).*step];

expectedNumCoors = length(timestamps);
hMissingSamps = nan;
if expectedNumCoors ~= length(xs)
    timestamps = [timestamps{:}];
    timeDiffs = diff(timestamps);
    
    if expectedNumCoors == 2*numSampsPerAxis-1
        warning('1 sample missing!');
        
        % The moment when we switched from z axis to x axis.
        [timeChangeAxis, idxChangeAxis] = max(timeDiffs);
        expTimePerSamp = median(timeDiffs);
        assert(timeChangeAxis>5*expTimePerSamp);
        timeDiffs(idxChangeAxis) = expTimePerSamp;
        
        % Find the most time consuming sample and if it takes longer than
        % 1.5 the time expected, we conclude there is a sample missing
        % right before it. Otherwise, conclude the missing sample is at the
        % end of all the measurements.
        %  - Note: this won't correct the cases:
        %    #1, #numSampsPerAxis, #(numSampsPerAxis+1)
        [maxTimeDiff, idxMaxTimeDiff] = max(timeDiffs);
        if maxTimeDiff>1.5*expTimePerSamp
            indicesMissingSamps = idxMaxTimeDiff+1;
        else
            indicesMissingSamps = 2*numSampsPerAxis;
        end
    elseif expectedNumCoors == 2*numSampsPerAxis-2
        warning('2 samples missing!');
        
        % The moment when we switched from z axis to x axis.
        [timeChangeAxis, idxChangeAxis] = max(timeDiffs);
        expTimePerSamp = median(timeDiffs);
        assert(timeChangeAxis>5*expTimePerSamp);
        timeDiffs(idxChangeAxis) = expTimePerSamp;
        
        % Find the most 2 time consuming sample and if any of them takes
        % longer than 1.5 the time expected, we conclude there is a sample
        % missing right before it. Otherwise, conclude any missing sample
        % is at the end of all the measurements.
        %  - Note: this won't correct some cases.
        [maxTimeDiff1, idxMaxTimeDiff1] = max(timeDiffs);
        timeDiffs(idxMaxTimeDiff1) = expTimePerSamp;
        [maxTimeDiff2, idxMaxTimeDiff2] = max(timeDiffs);
        assert(maxTimeDiff1>=maxTimeDiff2, ...
            'The first diffTime should be >= the second one!');
        if maxTimeDiff2>1.5*expTimePerSamp
            indicesMissingSamps = sort([idxMaxTimeDiff1+1;idxMaxTimeDiff2+1]);
            % Need to account for the first missing sample for the second
            % one.
            indicesMissingSamps(2) = indicesMissingSamps(2)+1;
        elseif maxTimeDiff1>1.5*expTimePerSamp
            indicesMissingSamps = [idxMaxTimeDiff1+1;2*numSampsPerAxis];
        else
            indicesMissingSamps = [2*numSampsPerAxis-1;2*numSampsPerAxis];
        end
    elseif expectedNumCoors == numSampsPerAxis/2
        % If half the samples are missing, then we will just assign these
        % half.
        warning('Half of the samples missing!');
        indicesMissingSamps = (numSampsPerAxis/2+1):numSampsPerAxis;
    else
        error([num2str(expectedNumCoors - length(xs)), ...
            ' samples missing! This is currently not supported!']);
    end
    
    % For debugging.
    newTs = timestamps;
    for idxMiss = indicesMissingSamps
        newTs = [newTs(1:(idxMiss-1)), nan, newTs(idxMiss:end)];
    end
    % Plot
    hMissingSamps = figure; title('Missing Sample for Plus Pattern');
    xlabel('Sample Index'); ylabel('Timestamps'); hold on;
    hInputTs = plot(timestamps,'rx'); 
    hNewTs = plot(newTs,'bo'); 
    hold off; legend([hInputTs, hNewTs], ...
        'Inpute Timestamps', 'Adjusted Timestamps'); 
    
    xs(indicesMissingSamps) = [];
    zs(indicesMissingSamps) = [];
end

end
% EOF