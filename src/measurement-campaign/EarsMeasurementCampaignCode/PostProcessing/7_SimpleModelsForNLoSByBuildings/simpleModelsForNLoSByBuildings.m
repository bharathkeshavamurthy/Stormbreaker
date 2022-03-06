% SIMPLEMODELSFORNLOSBYBUILDINGS We will try the KED model and dBLoss/m
% model for buildings, using the data from the continuous track on Holloway
% road.
%
% Yaguang Zhang, Purdue, 11/07/2017

clear; clc; close all;

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
    'PostProcessingResults', 'SimpleModelsForNLoSByBuildings');

% Reuse results from loadMeasCampaignInfo.m,
% evalPathLossesForContiTracks.m.
ABS_PATH_TO_TX_INFO_LOGS_FILE= fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation', 'txInfoLogs.mat');
ABS_PATH_TO_CONTI_PATH_LOSSES_FILE= fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputationConti', ...
    'contiPathLossesWithGpsInfo.mat');

% Signal frequency in Hz.
fInHz = 28.*10.^9;

% The time stamp to the continuous track on Holloway road.
TRACK_TIMESTAMP = '1497999912';

% Set this to be true to plot each GPS point on map one by one for manully
% inspection.
FLAG_MANUAL_INSPECTION = false;

% Set this to be true input to generate debug plots when the blockage
% points are computed.
FLAG_GEN_DEBUG_PLOTS_KED = false;

%% Before Processing the Data

disp(' -------------------------------- ')
disp('  simpleModelsForNLoSByBuildings ')
disp(' -------------------------------- ')

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

%% Extract the Data on Holloway Road

disp(' ')
disp('    Extracting data for the continuous track on Holloway road... ')

idxHollowayContiTrack = find(cellfun(@(relPath) ...
    contains(relPath, TRACK_TIMESTAMP), ...
    contiOutFilesRelPathsUnderDataFolder));
pathLossesWithGpsHol = contiPathLossesWithGpsInfo{idxHollowayContiTrack};

% Hard-coded info gotten from manual inspection.

% The lat and lon for the southeastern edge of Rickover (top view).
edgeLatLonsSERickover = [38.984588, -76.484921; 38.985053, -76.484444];
% The lat and lon for the northeastern edge of Michelson (top view).
edgeLatLonsNEMichelson = [38.984116, -76.484787; 38.983739, -76.484202];

% Heights in meters.
%   TX_HEIGHT_M is already loaded.
% Estimated height (in m) for Michelson Hall and Rickover Hall.
heightMMichelson = 20;
heightMRickover = 20;
% Estimated RX height in m.
rxHeightM = 1.5;

% We will generate the cross points (lat, lon) automatically.
[numGpsSamps, ~] = size(pathLossesWithGpsHol);
crossCoorsWithMichelson = nan(numGpsSamps,2);
crossCoorsWithRickover = nan(numGpsSamps,2);

disp('    Done!')

%% Manual Inspection

% All the NLoS sites.
disp(' ')
disp('    GPS info:')
disp(['        Path - ', ...
    cell2mat(contiOutFilesRelPathsUnderDataFolder{idxHollowayContiTrack})])

xsRick = edgeLatLonsSERickover(:,2);
ysRick = edgeLatLonsSERickover(:,1);
xsMich = edgeLatLonsNEMichelson(:,2);
ysMich = edgeLatLonsNEMichelson(:,1);
if FLAG_MANUAL_INSPECTION
    figure; hold on;
    for idx = 1:numGpsSamps
        curPathLossWithGps = pathLossesWithGpsHol(idx,:);
        disp(['SampleIndex = ', num2str(idx), ...
            ', latM = ', num2str(curPathLossWithGps(2)), ...
            ', lonM = ', num2str(curPathLossWithGps(3)), ...
            ', pathLossDb = ', num2str(curPathLossWithGps(1))])
        % Conti track.
        plot(curPathLossWithGps(3), curPathLossWithGps(2), ...
            '*', 'Color', ones(1,3)*0.9);
        % TX.
        plot(TX_LON, TX_LAT, '^b');
        % Building edges.
        plot(xsRick, ysRick, 'k.--');
        plot(xsMich, ysMich, 'k.-.');
    end
    disp(' ')
    
    plot_google_map('MapType', 'satellite');
    hGpsSamp = nan;
    hLink = nan;
    hXRick = nan;
    hXMich = nan;
    for idx = 1:numGpsSamps
        deleteHandlesInCell({hGpsSamp, hLink, hXRick, hXMich});
        
        curPathLossWithGps = pathLossesWithGpsHol(idx,:);
        hGpsSamp = plot(curPathLossWithGps(3), curPathLossWithGps(2), 'r*');
        % Find intersections.
        xsLink = [TX_LON, curPathLossWithGps(3)];
        ysLink = [TX_LAT, curPathLossWithGps(2)];
        hLink = plot(xsLink, ysLink, 'b-.');
        
        [crossLon, crossLat] = polyxpoly(xsLink,ysLink,xsMich,ysMich);
        if ~isempty(crossLon)
            assert(length(crossLon) <= 1, ...
                'There should be at most 1 intersection point (Rickover).');
            crossCoorsWithMichelson(idx,:) = [crossLat, crossLon];
            hXMich = plot(crossLon, crossLat, 'xr', 'MarkerSize', 7, 'LineWidth', 2);
        end
        
        [crossLon, crossLat] = polyxpoly(xsLink,ysLink,xsRick,ysRick);
        if ~isempty(crossLon)
            assert(length(crossLon) <= 1, ...
                'There should be at most 1 intersection point (Rickover).');
            crossCoorsWithRickover(idx,:) = [crossLat, crossLon];
            hXRick = plot(crossLon, crossLat, 'xr', 'MarkerSize', 7, 'LineWidth', 2);
        end
        
        title(['SampleIndex = ', num2str(idx)], 'Interpreter', 'none');
        disp('Press any key to continue...')
        pause;
    end
else
    for idx = 1:numGpsSamps
        curPathLossWithGps = pathLossesWithGpsHol(idx,:);
        % Find intersections.
        xsLink = [TX_LON, curPathLossWithGps(3)];
        ysLink = [TX_LAT, curPathLossWithGps(2)];
        
        [crossLon, crossLat] = polyxpoly(xsLink,ysLink,xsMich,ysMich);
        if ~isempty(crossLon)
            assert(length(crossLon) <= 1, ...
                'There should be at most 1 intersection point (Rickover).');
            crossCoorsWithMichelson(idx,:) = [crossLat, crossLon];
        end
        
        [crossLon, crossLat] = polyxpoly(xsLink,ysLink,xsRick,ysRick);
        if ~isempty(crossLon)
            assert(length(crossLon) <= 1, ...
                'There should be at most 1 intersection point (Rickover).');
            crossCoorsWithRickover(idx,:) = [crossLat, crossLon];
        end
    end
end

disp('Done!')
%% Use the KED Model

disp(' ')
disp('    Applying KED model ...')

knifeEdgeLossesKed = zeros(numGpsSamps,1);
lambdaInM = physconst('LightSpeed')./(fInHz);
% 3D distance in m between the TX and the RX.
dsInMFromTx = nan(numGpsSamps,1);
% Measured path losses in dB.
pathLossesDb = nan(numGpsSamps,1);
% idx = 90 is a good point to see the change of dominant knife-edge path.
% idx = 150 is a good point to see the blockage of the Rickover hall.
for idx = 1:numGpsSamps
    lossDbRick = 0;
    lossDbMich = 0;
    disp(['        Idx:', num2str(idx),'/', num2str(numGpsSamps), '...'])
    if all(~isnan(crossCoorsWithMichelson(idx,:)))
        % Note the result is negative.
        lossDbMich = computeKnifeEdgeLossDbKed( ...
            [TX_LAT, TX_LON], TX_HEIGHT_M, ...
            edgeLatLonsNEMichelson, ...
            crossCoorsWithMichelson(idx,:), heightMMichelson, ...
            [pathLossesWithGpsHol(idx,2), pathLossesWithGpsHol(idx,3)], ...
            rxHeightM, lambdaInM, FLAG_GEN_DEBUG_PLOTS_KED);
    end
    if all(~isnan(crossCoorsWithRickover(idx,:)))
        lossDbRick = computeKnifeEdgeLossDbKed( ...
            [TX_LAT, TX_LON], TX_HEIGHT_M, ...
            edgeLatLonsSERickover, ...
            crossCoorsWithRickover(idx,:), heightMRickover, ...
            [pathLossesWithGpsHol(idx,2), pathLossesWithGpsHol(idx,3)], ...
            rxHeightM, lambdaInM, FLAG_GEN_DEBUG_PLOTS_KED);
    end
    
    assert( ~all([lossDbRick lossDbMich]~=0), ...
        'At most one building could block the signal.');
    % A positive number.
    knifeEdgeLossesKed(idx) = -min([lossDbRick lossDbMich]);
    
    curPathLossWithGps = pathLossesWithGpsHol(idx,:);
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
pathLossInDbGaussianKed = pathLossInDbGaussianRef;
% Shift the path loss means by the sharp-knife model results.
for idxSamp = 1:numGpsSamps
    pathLossInDbGaussianKed(idxSamp).pathLossInDbMean = ...
        pathLossInDbGaussianKed(idxSamp).pathLossInDbMean ...
        + knifeEdgeLossesKed(idxSamp);
end

% Compute the RMSE.
[ ~, ~, rmseAggItu ] = computeRmseOverDist(dsInMFromTx, ...
    [pathLossInDbGaussianRef.pathLossInDbMean]', pathLossesDb);
[ ~, ~, rmseAggKed ] = computeRmseOverDist(dsInMFromTx, ...
    [pathLossInDbGaussianKed.pathLossInDbMean]', pathLossesDb);

% Save results to a .mat file.
pathKnifeEdgeLossesFileToSave = fullfile(ABS_PATH_TO_SAVE_PLOTS, ...
    'kifeEdgeLosses.mat');
save(pathKnifeEdgeLossesFileToSave, ...
    'TRACK_TIMESTAMP', 'fInHz', 'lambdaInM', ...
    'TX_LAT', 'TX_LON', 'TX_HEIGHT_M', 'rxHeightM', ...
    'heightMMichelson', 'heightMRickover', ...
    'crossCoorsWithRickover', 'crossCoorsWithMichelson', ...
    'knifeEdgeLossesKed', ...
    'dsInMFromTx', 'pathLossesDb', ...
    'pathLossInDbGaussianKed', 'pathLossInDbGaussianRef', ...
    'rmseAggKed', 'rmseAggItu');

disp('    Done!')

%% Plot
% We will plot pathLossesDb, pathLossInDbGaussianKed,
% pathLossInDbGaussianRef in one figure over dsInMFromTx.

% For plotting, need to order the points by distance between the TX and the
% RX.
[~, indicesToOrderDs] = sort(dsInMFromTx);
dsInMFromTxOrd = dsInMFromTx(indicesToOrderDs);

pathLossesDbOrd = pathLossesDb(indicesToOrderDs);
pathLossInDbGaussianKedOrd = pathLossInDbGaussianKed(indicesToOrderDs);
pathLossInDbGaussianRefOrd = pathLossInDbGaussianRef(indicesToOrderDs);

% Path loss over distance.
hPathLossOverDist = figure; hold on; colormap jet;
% Our basic transission losses.
yPlaneZerosPadding = zeros(length(dsInMFromTxOrd),1);
%plot3k([dsInMFromTxOrd, yPlaneZerosPadding, pathLossesDb], 'Marker', {'.', 10});
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
% ITU model with the sharp-knife results;
lineWidthShiftedItu = 1.5;
hMean = plot3(dsInMFromTxOrd, yPlaneOnesPadding, ...
    [pathLossInDbGaussianKedOrd.pathLossInDbMean]', 'k-', ...
    'LineWidth', lineWidthShiftedItu);
h3Sigma = plot3(dsInMFromTxOrd, yPlaneOnesPadding, ...
    [pathLossInDbGaussianKedOrd.pathLossInDbMean]' ...
    + 3.*[pathLossInDbGaussianKedOrd.pathLossInDbVar]', '-.', ...
    'Color', ones(1,3)*0.5, 'LineWidth', lineWidthShiftedItu);
plot3(dsInMFromTxOrd, yPlaneOnesPadding, ...
    [pathLossInDbGaussianKedOrd.pathLossInDbMean]' ...
    - 3.*[pathLossInDbGaussianKedOrd.pathLossInDbVar]', '-.', ...
    'Color', ones(1,3)*0.5, 'LineWidth', lineWidthShiftedItu);
% Put the aggregated root meas square errors on the plot.
text(257, 0, 167,...
    {'Root Mean Square Errors', ...
    ['  For Ref ITU Mean: ', num2str(rmseAggItu, '%.2f'), ' dB'], ...
    ['  For ITU with KED Mean: ', num2str(rmseAggKed, '%.2f'), ' dB']}, ...
    'Rotation', 0, ...
    'Color', 'k', 'HorizontalAlignment', 'left', ...
    'VerticalAlignment', 'top', 'Interpreter', 'none', 'FontSize', 11);
view(0, 0);
set(gca, 'XScale', 'log'); grid on;
newXTicks = [250, 260, 270, 280, 290, 300, 320, 340, 360, 380];
set(gca, 'XTickLabels',newXTicks);
set(gca, 'XTick',newXTicks);
hLegend = legend([hMeas, hItu3SigmaRange, hMean, h3Sigma], ...
    'Measurements', ...
    'Ref ITU 3 sigma range', 'ITU with KED Mean', '3 sigma offsets', ...
    'Location','southeast');
% transparentizeCurLegends;
xlabel('Distance to TX (m) in Log Scale'); ylabel(''); 
zlabel('Path Loss (dB)'); set(gcf, 'Color', 'w');
set(gca, 'FontWeight', 'bold'); %grid minor;
hold off;

% Save plots.
absPathToSavePlots = fullfile(ABS_PATH_TO_SAVE_PLOTS, 'pathLossOverDist');
saveas(hPathLossOverDist, [absPathToSavePlots, '.fig']);
saveas(hPathLossOverDist, [absPathToSavePlots, '.png']);

% EOF