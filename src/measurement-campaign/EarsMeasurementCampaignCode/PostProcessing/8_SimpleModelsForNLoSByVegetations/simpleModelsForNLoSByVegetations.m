% SIMPLEMODELSFORNLOSBYVEGETATIONS We will use a fixed path loss in dB/m of
% foliage to model the blockages caused by the trees, using the data from
% the continuous track on Upshur road.
%
% Yaguang Zhang, Purdue, 11/13/2017

% Checking the existance of dBPerM for minimizeRmseOverDbPerM.
if ~exist('dBPerM', 'var')
    clearvars -except dBPerM RMSEs idxTest; clc;
    flagMakePlots = true;
else
    flagMakePlots = false;
end

close all;

%% Configurations

warning('on');

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd('..');
% We also need the function ituSiteGeneralOverRoofTopsLoS.m.
addpath(fullfile(pwd, '5_ComparePathLossesWithItuModels'));
setPath;

% Configure other paths accordingly.
ABS_PATH_TO_SAVE_PLOTS = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'SimpleModelsForNLoSByVegetations');

% Reuse results from loadMeasCampaignInfo.m,
% evalPathLossesForContiTracks.m.
ABS_PATH_TO_TX_INFO_LOGS_FILE= fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation', 'txInfoLogs.mat');
ABS_PATH_TO_CONTI_PATH_LOSSES_FILE= fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputationConti', ...
    'contiPathLossesWithGpsInfo.mat');

% Signal frequency in Hz.
fInHz = 28.*10.^9;

% The time stamp to the continuous track on Upshur road.
TRACK_TIMESTAMP = '1497981683';

% Set this to be true to plot each GPS point on map one by one for manully
% inspection.
FLAG_MANUAL_INSPECTION = false;

% The model parameter for the path loss caused by foliage. By checking the
% existance of dBPerM first, we can automate the process of finding dBPerM
% to minimize RMSE.
if ~exist('dBPerM', 'var')
    dBPerM = 0.13;
end

%% Before Processing the Data

disp(' ---------------------------------- ')
disp('  simpleModelsForNLoSByVegetations ')
disp(' ---------------------------------- ')

% Create directories if necessary.
if exist(ABS_PATH_TO_SAVE_PLOTS, 'dir')~=7
    mkdir(ABS_PATH_TO_SAVE_PLOTS);
end

%% Get Info for Measurement Data Files and Path Losses

disp(' ')
disp('    Loading results from: ')
disp('      - loadMeasCampaignInfo.m')
disp('      - evalPathLossesForContiTracks.m')

assert(exist(ABS_PATH_TO_TX_INFO_LOGS_FILE, 'file')==2, ...
    'Couldn''t find txInfoLogs.mat! Please run PostProcessing/4_0_PathLossComputation/loadMeasCampaignInfo.m first.');
assert(exist(ABS_PATH_TO_CONTI_PATH_LOSSES_FILE, 'file')==2, ...
    'Couldn''t find contiPathLossesWithGpsInfo.mat! Please run PostProcessing/4_0_PathLossComputation/evalPathLossesForContiTracks.m first.');

% The data have been processed before and the result files have been found.
disp('    Found all .mat files required.');
disp('        Loading the results...')

% Get records of the TxInfo.txt files (among other contant parameters for
% the measurement campaign, e.g. F_S, TX_LAT, TX_LON, and TX_POWER_DBM):
% 'TX_INFO_LOGS' and 'TX_INFO_LOGS_ABS_PAR_DIRS'.
load(ABS_PATH_TO_TX_INFO_LOGS_FILE);
% Get 'contiPathLossesWithGpsInfo', 'contiOutFilesRelPathsUnderDataFolder',
% and 'contiOutFileIndicesReflection'.
load(ABS_PATH_TO_CONTI_PATH_LOSSES_FILE);

disp('    Done!')

%% Extract the Data on Upshur Road

disp(' ')
disp('    Extracting data for the continuous track on Upshur road... ')

idxHollowayContiTrack = find(cellfun(@(relPath) ...
    contains(relPath, TRACK_TIMESTAMP), ...
    contiOutFilesRelPathsUnderDataFolder));
pathLossesWithGpsUps = contiPathLossesWithGpsInfo{idxHollowayContiTrack};

% Hard-coded tree location information. We will model the foliage part of a
% tree as a sphere. Each row models a tree in terms of [lat, lon,
% folaigeRadiusInMeter, foliageCenterHeightInMeter].
%  largeTreeFoliageRM  = 10;   % Ref:  20 m diameter on Google map.
mediumTreeFoliageRM = 7;  % Ref: 13.5 m diameter on Google map.
smallTreeFoliageRM  = 5;  % Ref:  7.5 m diameter on Google map.

% For simplicity, we will use the radius as height, too.
%  largeTreeCenterHM  = largeTreeFoliageRM;
mediumTreeCenterHM = mediumTreeFoliageRM;
smallTreeCenterHM  = smallTreeFoliageRM;

% For simplicity, we will use the radius plus a constant as height.
%  largeTreeCenterHM  = largeTreeFoliageRM  + 2;
mediumTreeCenterHM = mediumTreeFoliageRM + 1.5;
smallTreeCenterHM  = smallTreeFoliageRM  + 1;

% Northest example trees: small, medium and large.
%     38.984553, -76.492457, smallTreeFoliageRM, smallTreeCenterHM; ...
%      38.984493, -76.492350, mediumTreeFoliageRM, mediumTreeCenterHM; ...
%     38.984369, -76.492154, largeTreeFoliageRM, largeTreeCenterHM; ...
trees = [...
    ... % Small trees.
    38.984489, -76.492224, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.984379, -76.492071, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.984322, -76.491968, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.984248, -76.491847, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.984182, -76.491763, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.984120, -76.491633, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.984048, -76.491517, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983969, -76.491417, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983889, -76.491295, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983825, -76.491176, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983744, -76.491067, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983680, -76.490959, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983605, -76.490858, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983526, -76.490733, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983382, -76.490523, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983223, -76.490287, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983153, -76.490159, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983097, -76.490060, smallTreeFoliageRM, smallTreeCenterHM; ...
    ... % Trees across the road.
    38.983266, -76.490131, mediumTreeFoliageRM, mediumTreeCenterHM; ...
    38.983358, -76.490236, mediumTreeFoliageRM, mediumTreeCenterHM; ...
    38.983673, -76.490761, mediumTreeFoliageRM, mediumTreeCenterHM; ...
    38.983750, -76.490866, mediumTreeFoliageRM, mediumTreeCenterHM; ...
    38.983813, -76.490973, mediumTreeFoliageRM, mediumTreeCenterHM; ...
    38.983876, -76.491081, mediumTreeFoliageRM, mediumTreeCenterHM; ...
    38.983960, -76.491203, mediumTreeFoliageRM, mediumTreeCenterHM; ...
    38.984295, -76.491711, mediumTreeFoliageRM, mediumTreeCenterHM; ...
    ... % Bushes across the road.
    38.983158, -76.489960, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983209, -76.490043, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983421, -76.490363, smallTreeFoliageRM, smallTreeCenterHM; ...
    38.983495, -76.490475, smallTreeFoliageRM, smallTreeCenterHM; ...
    
    ];

% Heights in meters.
%   TX_HEIGHT_M is already loaded.
% Estimated RX height in m.
rxHeightM = 1.5;

[numGpsSamps, ~] = size(pathLossesWithGpsUps);

disp('    Done!')

%% Manual Inspection

% All the NLoS sites.
disp(' ')
disp('    GPS info:')
disp(['        Path - ', ...
    cell2mat(contiOutFilesRelPathsUnderDataFolder{idxHollowayContiTrack})])

% We will compute the total foliage width for each GPS sample here.
totalWidthsInM = zeros(numGpsSamps,1);
if FLAG_MANUAL_INSPECTION
    figure; hold on;
    for idx = 1:numGpsSamps
        curPathLossWithGps = pathLossesWithGpsUps(idx,:);
        disp(['SampleIndex = ', num2str(idx), ...
            ', latM = ', num2str(curPathLossWithGps(2)), ...
            ', lonM = ', num2str(curPathLossWithGps(3)), ...
            ', pathLossDb = ', num2str(curPathLossWithGps(1))])
        % Conti track.
        plot(curPathLossWithGps(3), curPathLossWithGps(2), ...
            '*', 'Color', ones(1,3)*0.9);
        % TX.
        plot(TX_LON, TX_LAT, '^b');
        % Trees.
        hTrees = plot(trees(:,2), trees(:,1), 'ko', 'MarkerFaceColor', 'g');
    end
    disp(' ')
    
    plot_google_map('MapType', 'satellite');
    hGpsSamp = nan;
    hLink = nan;
    hIncidPts = nan;
    hExitPts = nan;
    for idx = 1:numGpsSamps
        deleteHandlesInCell({hGpsSamp, hLink, hIncidPts, hExitPts});
        
        curPathLossWithGps = pathLossesWithGpsUps(idx,:);
        hGpsSamp = plot(curPathLossWithGps(3), curPathLossWithGps(2), ...
            'k*', 'MarkerSize', 7, 'LineWidth', 2);
        
        xsLink = [TX_LON, curPathLossWithGps(3)];
        ysLink = [TX_LAT, curPathLossWithGps(2)];
        hLink = plot(xsLink, ysLink, 'b-.', 'LineWidth', 2);
        
        % Find intersections.
        [intersPtsXYZs, intersPtsLatLons, widthsInM, totalWidthsInM(idx)] ...
            = findIntersPtsForSphereTrees([TX_LAT, TX_LON], TX_HEIGHT_M, ...
            [curPathLossWithGps(2), curPathLossWithGps(3)], rxHeightM, ...
            trees);
        
        if totalWidthsInM(idx)>0
            disp(['    Foliage width ', num2str(totalWidthsInM(idx)), 'm'])
            hIncidPts = plot(intersPtsLatLons(:,2), intersPtsLatLons(:,1), ...
                'xr', 'MarkerSize', 7, 'LineWidth', 2);
            hExitPts = plot(intersPtsLatLons(:,4), intersPtsLatLons(:,3), ...
                'xy', 'MarkerSize', 7, 'LineWidth', 2);
            legend([hTrees, hLink, hGpsSamp, hIncidPts, hExitPts], ...
                'Trees', 'LoS Path', 'Rx', 'Incident Points', 'Exit Points');
        else
            legend([hTrees, hLink, hGpsSamp], ...
                'Trees', 'LoS Path', 'Rx');
        end
        
        title(['SampleIndex = ', num2str(idx)], 'Interpreter', 'none');
        disp('Press any key to continue...')
        pause;
    end
else
    for idx = 1:numGpsSamps
        curPathLossWithGps = pathLossesWithGpsUps(idx,:);
        % Find intersections.
        [intersPtsXYZs, intersPtsLatLons, widthsInM, totalWidthsInM(idx)] ...
            = findIntersPtsForSphereTrees([TX_LAT, TX_LON], TX_HEIGHT_M, ...
            [curPathLossWithGps(2), curPathLossWithGps(3)], rxHeightM, ...
            trees);
    end
end

disp('Done!')
%% Use the Vegetation Loss Model

disp(' ')
disp('    Applying path loss in dB per meter of foliage width model ...')

% 3D distance in m between the TX and the RX.
dsInMFromTx = nan(numGpsSamps,1);
% Measured path losses in dB.
pathLossesDb = nan(numGpsSamps,1);
vegLoss = nan(numGpsSamps,1);
for idx = 1:numGpsSamps
    vegLoss(idx) = totalWidthsInM(idx).*dBPerM;
    
    curPathLossWithGps = pathLossesWithGpsUps(idx,:);
    pathLossesDb(idx) = curPathLossWithGps(1);
    dsInMFromTx(idx) = norm( ....
        [1000.*lldistkm([curPathLossWithGps(2) curPathLossWithGps(3)], ...
        [TX_LAT,TX_LON]), TX_HEIGHT_M-rxHeightM]);
end

%% ITU LoS Over-Toproof Site General Model
% Also compute the ITU LoS results as the baseline.
fInGHz = fInHz .* (0.1^9);
pathLossInDbGaussianRef = arrayfun(@(d) ...
    ituSiteGeneralOverRoofTopsLoS( fInGHz, d ), dsInMFromTx);
pathLossInDbVegDbPerM = pathLossInDbGaussianRef;
% Shift the path loss means by the vegetation model results.
for idxSamp = 1:numGpsSamps
    pathLossInDbVegDbPerM(idxSamp).pathLossInDbMean = ...
        pathLossInDbVegDbPerM(idxSamp).pathLossInDbMean ...
        + vegLoss(idxSamp);
end

% Compute the RMSE.
[ ~, ~, rmseAggItu ] = computeRmseOverDist(dsInMFromTx, ...
    [pathLossInDbGaussianRef.pathLossInDbMean]', pathLossesDb);
[ ~, ~, rmseAggVegDbPerM ] = computeRmseOverDist(dsInMFromTx, ...
    [pathLossInDbVegDbPerM.pathLossInDbMean]', pathLossesDb);

% Save results to a .mat file.
pathVegDbPerMLossesFileToSave = fullfile(ABS_PATH_TO_SAVE_PLOTS, ...
    'vegDbPerMLosses.mat');
save(pathVegDbPerMLossesFileToSave, ...
    'TRACK_TIMESTAMP', 'fInHz', ...
    'TX_LAT', 'TX_LON', 'TX_HEIGHT_M', 'rxHeightM', ...
    'totalWidthsInM', 'dBPerM', ...
    'dsInMFromTx', 'pathLossesDb', ...
    'pathLossInDbVegDbPerM', 'pathLossInDbGaussianRef', ...
    'rmseAggVegDbPerM', 'rmseAggItu');

disp('    Done!')

%% Plot
% We will plot pathLossesDb, pathLossInDbVegDbPerM, pathLossInDbGaussianRef
% in one figure over dsInMFromTx.

if flagMakePlots
    % For plotting, need to order the points by distance between the TX and
    % the RX.
    [~, indicesToOrderDs] = sort(dsInMFromTx);
    dsInMFromTxOrd = dsInMFromTx(indicesToOrderDs);
    
    pathLossesDbOrd = pathLossesDb(indicesToOrderDs);
    pathLossInDbVegDbPerMOrd = pathLossInDbVegDbPerM(indicesToOrderDs);
    pathLossInDbGaussianRefOrd = pathLossInDbGaussianRef(indicesToOrderDs);
    
    % Compute windowed mean for the measured path losses for reference.
    windowSize = 11; % An odd interger.
    assert(rem(windowSize,1)==0 && mod(windowSize,2)==1 && windowSize>0, ...
        'Window size for computing the windowed means should be an odd positive integer!')
    halfWindowSize = (windowSize-1)/2;
    pathLossInDbWinMean = nan(numGpsSamps,1);
    for idxSamp = 1:numGpsSamps
        indicesInWindow = max([1, idxSamp-halfWindowSize]):1:min([numGpsSamps, idxSamp+halfWindowSize]);
        pathLossInDbWinMean(idxSamp) = mean(pathLossesDbOrd(indicesInWindow));
    end
    
    % Path loss over distance.
    hPathLossOverDist = figure; hold on; colormap jet;
    % Our basic transission losses.
    yPlaneZerosPadding = zeros(length(dsInMFromTx),1);
    %plot3k([dsInMFromTx, yPlaneZerosPadding, pathLossesDb], 'Marker',
    %{'.', 10});
    hMeas = plot3(dsInMFromTxOrd, yPlaneZerosPadding, pathLossesDbOrd, 'b.', ...
        'MarkerSize', 10);
    % Ref ITU model results.
    yPlaneOnesPadding = ones(length(dsInMFromTxOrd),1);
    ituPatchXs = [dsInMFromTxOrd; dsInMFromTxOrd(end:-1:1)];
    ituPatchYs = [yPlaneOnesPadding; yPlaneOnesPadding];
    ituRefBottomZs = [pathLossInDbGaussianRefOrd.pathLossInDbMean]' ...
        - 3.*[pathLossInDbGaussianRefOrd.pathLossInDbVar]';
    ituPatchZs = [[pathLossInDbGaussianRefOrd.pathLossInDbMean]' ...
        + 3.*[pathLossInDbGaussianRefOrd.pathLossInDbVar]'; ...
        ituRefBottomZs(end:-1:1)];
    hItu3SigmaRange = fill3(ituPatchXs,ituPatchYs,ituPatchZs, ...
        ones(1,3).*0.9, 'LineStyle', 'none');
    alpha(hItu3SigmaRange, 0.5);
    % ITU model with the vegetation results;
    hMean = plot3(dsInMFromTxOrd, yPlaneOnesPadding, ...
        [pathLossInDbVegDbPerMOrd.pathLossInDbMean]', 'k-', 'LineWidth', 1);
    h3Sigma = plot3(dsInMFromTxOrd, yPlaneOnesPadding, ...
        [pathLossInDbVegDbPerMOrd.pathLossInDbMean]' ...
        + 3.*[pathLossInDbVegDbPerMOrd.pathLossInDbVar]', '-.', ...
        'Color', ones(1,3)*0.5, 'LineWidth', 1);
    plot3(dsInMFromTxOrd, yPlaneOnesPadding, ...
        [pathLossInDbVegDbPerMOrd.pathLossInDbMean]' ...
        - 3.*[pathLossInDbVegDbPerMOrd.pathLossInDbVar]', '-.', ...
        'Color', ones(1,3)*0.5, 'LineWidth', 1);
    % Windowed mean.
    hWinMean = plot3(dsInMFromTxOrd, yPlaneOnesPadding, ...
        pathLossInDbWinMean, 'b-.', 'LineWidth', 1);
    % Put the aggregated root meas square errors on the plot.
    text(412, 0, 107,...
        {'Root Mean Square Errors:', ...
        ['  Ref ITU Mean ', num2str(rmseAggItu, '%.2f'), ' dB'], ...
        ['  ITU with Veg. Mean ', num2str(rmseAggVegDbPerM, '%.2f'), ' dB']}, ...
        'Rotation', 0, ...
        'Color', 'k', 'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top', 'Interpreter', 'none', 'FontSize', 10);
    view(0, 0);
    set(gca, 'XScale', 'log'); grid on;
    newXTicks = [320, 340, 360, 380, 400, 420, 440, 460, 480];
    set(gca, 'XTickLabels',newXTicks);
    set(gca, 'XTick',newXTicks);
    hLegend = legend([hMeas, hWinMean, hItu3SigmaRange, hMean, h3Sigma], ...
        'Measurements', 'Measurement Windowed Mean', ...
        'Ref ITU 3 sigma range', 'ITU with Veg. Mean', '3 sigma offsets', ...
        'Location','northwest');
    % transparentizeCurLegends;
    xlabel('Distance to Tx (m) in log scale'); ylabel('');
    zlabel('Path Loss (dB)');
    hold off;
    
    % Plot the foliage width in another figure for reference.
    hEstiFoliageWidths = figure; hold on;
    plot(dsInMFromTxOrd, totalWidthsInM(indicesToOrderDs), 'k-.');
    plot(dsInMFromTxOrd, totalWidthsInM(indicesToOrderDs), 'b*');
    hold off; set(gca, 'XScale', 'log'); grid on;
    newXTicks = [320, 340, 360, 380, 400, 420, 440, 460, 480];
    set(gca, 'XTickLabels',newXTicks);
    set(gca, 'XTick',newXTicks);
    xlabel('Distance to Tx (m) in log scale'); ylabel('Esitimated foliage width');
    
    % Save plots.
    absPathToSavePlots = fullfile(ABS_PATH_TO_SAVE_PLOTS, 'pathLossOverDist');
    saveas(hPathLossOverDist, [absPathToSavePlots, '.fig']);
    saveas(hPathLossOverDist, [absPathToSavePlots, '.png']);
    
    % Save plots.
    absPathToSavePlots = fullfile(ABS_PATH_TO_SAVE_PLOTS, 'estiFoliageWidthOverDist');
    saveas(hEstiFoliageWidths, [absPathToSavePlots, '.fig']);
    saveas(hEstiFoliageWidths, [absPathToSavePlots, '.png']);
    
end

clearvars dBPerM;

% EOF