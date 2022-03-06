% PLOTBASICTRANSLOSSESBYCATEGORY Plot by Tx environment category (defined
% by the .txt files in the Categories folder alongside this .m file) the
% antenna-independent path losses (Basic Transmission Losses) gotten via
% evalPathLosses.m.
%
% Yaguang Zhang, Purdue, 10/06/2017

clear; clc; close all; dbstop if error;

%% Configurations

warning('on');

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
% We will need later the category folder for .txt files.
ABS_PATH_TO_CATEGORY_TXTS = fullfile(pwd, 'Categories');
cd('..'); setPath;

% Configure other paths accordingly.
ABS_PATH_TO_SAVE_PLOTS = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PlotBasicTransLossesByCategory');

% Reuse results from evalPathLosses.m and loadMeasCampaignInfo.m.
ABS_PATH_TO_PATH_LOSSES_FILE= fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation', ...
    'pathLossesWithGpsInfo.mat');
ABS_PATH_TO_TX_INFO_LOGS_FILE= fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation', 'txInfoLogs.mat');

% For setting the threshold during the noise elimination.
NUM_SIGMA_FOR_THRESHOLD = 3.5;

%% Before Processing the Data

disp(' -------------------------------- ')
disp('  plotBasicTransLossesByCategory')
disp(' -------------------------------- ')

% Create directories if necessary.
if exist(ABS_PATH_TO_SAVE_PLOTS, 'dir')~=7
    mkdir(ABS_PATH_TO_SAVE_PLOTS);
end

%% Get Info for Measurement Data Files and Calibration Polynomials

disp(' ')
disp('    Loading results from: ')
disp('      - pathLossesWithGpsInfo.m')

assert(exist(ABS_PATH_TO_PATH_LOSSES_FILE, 'file')==2, ...
    'Couldn''t find txInfoLogs.mat! Please run PostProcessing/4_0_PathLossComputation/evalPathLosses.m first.');
assert(exist(ABS_PATH_TO_TX_INFO_LOGS_FILE, 'file')==2, ...
    'Couldn''t find txInfoLogs.mat! Please run PostProcessing/4_0_PathLossComputation/loadMeasCampaignInfo.m first.');

% The data have been processed before and the result files have been found.
disp('    Found all .mat files required.');
disp('        Loading the results...')
% Get 'pathLossesWithGpsInfo', and 'relPathsOutFilesUnderDataFolder'. Note
% that these path losses are actually the Basic Transmission Losses we
% need.
load(ABS_PATH_TO_PATH_LOSSES_FILE);
load(ABS_PATH_TO_TX_INFO_LOGS_FILE);

disp('    Done!')

%% Load the Category Files

disp(' ')
disp('    Loading category .txt files ...')

catTxtsDirs = rdir(fullfile(ABS_PATH_TO_CATEGORY_TXTS, '*.txt'), ...
    '', false);
catTxts = arrayfun(@(d) loadCategoryTxt(d.name), catTxtsDirs);

disp('    Done!')

%% Plot

disp(' ')
disp('    Plotting...')

% Replace '/' and '\' with '_' for relPathsOutFilesUnderDataFolder.
seriesLabelsOutFiles = cellfun(@(p) ...
    strrep(strrep( fileparts(p), '\', '_' ),'/','_'), ...
    relPathsOutFilesUnderDataFolder, 'UniformOutput', false);

numCategories = length(catTxts);
for idxCat = 1:numCategories
    curCatTxt = catTxts(idxCat);
    
    disp(['        Category ', num2str(idxCat), '/', ...
        num2str(numCategories), ': ', curCatTxt.category])
    
    boolsRelOutFiles = cellfun(@(s) ...
        any(cellfun(@(sNeeded) strcmp(s, sNeeded), curCatTxt.series)), ...
        seriesLabelsOutFiles);
    if sum(boolsRelOutFiles) == 0
        warning(['No path loss results found for category: ', ...
            curCatTxt.category]);
        continue;
    end
    curPathLossesWithGpsInfo = pathLossesWithGpsInfo(boolsRelOutFiles,:);
    
    boolsInvalidCoor = curPathLossesWithGpsInfo(:,2)==0 ...
        & curPathLossesWithGpsInfo(:,3)==0;
    if any(boolsInvalidCoor)
        warning([num2str(sum(boolsInvalidCoor)), ...
            ' invalid (lat, lon) pairs detected (both are 0).', ...
            ' We will ignore these points together with their path losses.']);
    end
    pathLossesWithValidGps = curPathLossesWithGpsInfo(~boolsInvalidCoor,:);
    
    boolsInfPathloss = isinf(pathLossesWithValidGps(:,1));
    if any(boolsInfPathloss)
        warning([num2str(sum(boolsInfPathloss)), ...
            ' inf path loss detected.', ...
            ' We will show these points at z=0 with different markers.']);
    end
    validPathLossesWithValidGps = pathLossesWithValidGps(~boolsInfPathloss,:);
    
    % Plot path losses on map with individual GPS coordinates.
    hBasicTransLossesOnMapIndi = figure; hold on; colormap jet;
    plot(validPathLossesWithValidGps(:,3), validPathLossesWithValidGps(:,2), 'w.');
    plot(pathLossesWithValidGps(boolsInfPathloss,3), ...
        pathLossesWithValidGps(boolsInfPathloss,2), 'kx');
    hTx = plot(TX_LON, TX_LAT, '^w', 'MarkerFaceColor', 'b');
    plot_google_map('MapType','satellite');
    plot3k([validPathLossesWithValidGps(:,3), validPathLossesWithValidGps(:,2), ...
        validPathLossesWithValidGps(:,1)], 'Marker', {'.', 12});
    % The command plot_google_map messes up the color legend of plot3k, so
    % we will have to fix it here.
    hCb = findall( allchild(hBasicTransLossesOnMapIndi), 'type', 'colorbar');
    hCb.Ticks = linspace(1,length(colormap),length(hCb.TickLabels));
    hold off; grid on; view(0, 90); legend(hTx, 'TX');
    title(['Path Losses on Map - ', curCatTxt.category]);
    xlabel('Lon'); ylabel('Lat'); zlabel('Path Loss (dB)');
    
    % Plot path losses on map with average GPS coordinates.
    hBasicTransLossesOnMap = figure; hold on; colormap jet;
    plot(validPathLossesWithValidGps(:,5), validPathLossesWithValidGps(:,4), 'w.');
    plot(pathLossesWithValidGps(boolsInfPathloss,3), ...
        pathLossesWithValidGps(boolsInfPathloss,2), 'kx');
    hTx = plot(TX_LON, TX_LAT, '^w', 'MarkerFaceColor', 'b');
    plot_google_map('MapType','satellite');
    plot3k([validPathLossesWithValidGps(:,5), validPathLossesWithValidGps(:,4), ...
        validPathLossesWithValidGps(:,1)], 'Marker', {'.', 12});
    % The command plot_google_map messes up the color legend of plot3k, so
    % we will have to fix it here.
    hCb = findall( allchild(hBasicTransLossesOnMap), 'type', 'colorbar');
    hCb.Ticks = linspace(1,length(colormap),length(hCb.TickLabels));
    hold off; grid on; view(45, 45); legend(hTx, 'TX');
    title(['Path Losses on Map - Averaged Site Locations - ', curCatTxt.category]);
    xlabel('Lon'); ylabel('Lat'); zlabel('Path Loss (dB)');
    
    % Plot path losses over distance from Tx.
    validPLWithValidGPSCell = num2cell(validPathLossesWithValidGps, 2);
    distsFromTx = cellfun(@(s) 1000.*lldistkm([s(2) s(3)],[TX_LAT,TX_LON]), ...
        validPLWithValidGPSCell);
    
    hBasicTransLossesOverDist = figure; hold on; colormap jet;
    plot3k([distsFromTx, zeros(length(distsFromTx),1), ...
        validPathLossesWithValidGps(:,1)], 'Marker', {'.', 6});
    curAxis = axis;
    axis([min(distsFromTx)-10, max(distsFromTx)+100, curAxis(3:6)]);
    view(0, 0); set(gca, 'XScale', 'log'); grid on;
    newXTicks = [10,100,200,500,1000];
    set(gca, 'XTickLabels',newXTicks);
    set(gca, 'XTick',newXTicks);
    title(['Path Losses over Distance ', curCatTxt.category]);
    xlabel('Distance to Tx (m)'); ylabel(''); zlabel('Path Loss (dB)');
    
    % Save the plots.
    pathToSavePlotsWithCatPrefix = fullfile(ABS_PATH_TO_SAVE_PLOTS, ...
        curCatTxt.category);
    pathPathossesOnMapIndiFileToSave ...
        = strcat(pathToSavePlotsWithCatPrefix, ...
        '-basicTransLossesOnMapIndi');
    saveas(hBasicTransLossesOnMapIndi, [pathPathossesOnMapIndiFileToSave, '.png']);
    saveas(hBasicTransLossesOnMapIndi, [pathPathossesOnMapIndiFileToSave, '.fig']);
    pathPathossesOnMapFileToSave ...
        = strcat(pathToSavePlotsWithCatPrefix, ...
        '-basicTransLossesOnMap');
    saveas(hBasicTransLossesOnMap, [pathPathossesOnMapFileToSave, '.png']);
    saveas(hBasicTransLossesOnMap, [pathPathossesOnMapFileToSave, '.fig']);
    pathBasicTransLossesOverDistFileToSave ...
        = strcat(pathToSavePlotsWithCatPrefix, ...
        '-basicTransLossesOverDist');
    saveas(hBasicTransLossesOverDist, [pathBasicTransLossesOverDistFileToSave, '.png']);
    saveas(hBasicTransLossesOverDist, [pathBasicTransLossesOverDistFileToSave, '.fig']);
    
    close all;
end

disp('    Done!')

% EOF