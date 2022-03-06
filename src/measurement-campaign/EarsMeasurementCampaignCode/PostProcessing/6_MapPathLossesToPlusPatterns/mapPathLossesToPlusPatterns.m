% MAPPATHLOSSESTOPLUSPATTERNS This script will map the large-scale and SIMO
% measurements at each site back to the "+".
%
% Yaguang Zhang, Purdue, 10/17/2017

clear; clc; close all;

%% Configurations

warning('on');

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd('..'); setPath;

% Configure other paths accordingly.
ABS_PATH_TO_SAVE_PLOTS = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'MapPathLossesToPlusPatterns');

% Reuse results from loadMeasCampaignInfo.m, evalPathLosses.m.
ABS_PATH_TO_TX_INFO_LOGS_FILE= fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation', 'txInfoLogs.mat');
ABS_PATH_TO_PATH_LOSSES_FILE= fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'PathLossComputation', ...
    'pathLossesWithGpsInfo.mat');

%% Before Processing the Data

disp(' ----------------------------- ')
disp('  mapPathLossesToPlusPatterns ')
disp(' ----------------------------- ')

% Create directories if necessary.
if exist(ABS_PATH_TO_SAVE_PLOTS, 'dir')~=7
    mkdir(ABS_PATH_TO_SAVE_PLOTS);
end

%% Get Info for Measurement Data Files and Calibration Polynomials

disp(' ')
disp('    Loading results from: ')
disp('      - loadMeasCampaignInfo.m')
disp('      - evalPathLosses.m')

assert(exist(ABS_PATH_TO_TX_INFO_LOGS_FILE, 'file')==2, ...
    'Couldn''t find txInfoLogs.mat! Please run PostProcessing/4_0_PathLossComputation/loadMeasCampaignInfo.m first.');
assert(exist(ABS_PATH_TO_PATH_LOSSES_FILE, 'file')==2, ...
    'Couldn''t find pathLossesWithGpsInfo.mat! Please run PostProcessing/4_0_PathLossComputation/evalPathLosses.m first.');

% The data have been processed before and the result files have been found.
disp('    Found all .mat files required.');
disp('        Loading the results...')

% Get records of the TxInfo.txt files (among other contant parameters for
% the measurement campaign, e.g. F_S, TX_LAT, TX_LON, and TX_POWER_DBM):
% 'TX_INFO_LOGS' and 'TX_INFO_LOGS_ABS_PAR_DIRS'.
load(ABS_PATH_TO_TX_INFO_LOGS_FILE);
% Get 'pathLossesWithGpsInfo', 'relPathsOutFilesUnderDataFolder', and
% 'maxMeasurablePathLossInfo'.
load(ABS_PATH_TO_PATH_LOSSES_FILE);

disp('    Done!')

%% Map the Measurements from Each Site Back to the "+" Pattern
% We will have a right-hand coordinate system of (x, z) in meter where the
% RX antenna points at +y. More specificially, +x is the right-hand side of
% the antenna and +z is the top of it. (0, 0) is the home position (x at
% center, z at top) of the platform. The resulted (x, z) for each
% measurement will be appended to pathLossesWithGpsInfo.

disp(' ')
disp('    Mapping LoS measurements ...')

% Path losses with GPS information as well as the plus pattern coordinates.
[numRowRecords, ~] = size(pathLossesWithGpsInfo);
pathLossesWithGpsAndPlusCoor ...
    = [pathLossesWithGpsInfo, nan(numRowRecords, 2)];
% Locate all measurements for each site.
counterRowRecord = 0;
% Only process the data that are not listed in relPathSegInvalidData.
[~, ~, ~, boolsInvalidData] ...
    = checkValidityOfPathLossesWithGpsInfo(pathLossesWithGpsInfo, ...
    relPathsOutFilesUnderDataFolder);
while counterRowRecord < numRowRecords
    idxNextRow = counterRowRecord+1;
    if boolsInvalidData(idxNextRow)
        counterRowRecord = counterRowRecord+1;
    else
        % Current site info.
        curSiteRelPath = relPathsOutFilesUnderDataFolder{idxNextRow};
        [ date, type, serNum, timestamp ] ...
            = parseOutFileRelPath( curSiteRelPath );
        
        % Via the relative path for the parent directory, find all the
        % measurements for this site.
        curSiteRelPath = fullfile([date,'_',type], ['Series_',num2str(serNum)]);
        % All path loss records for this site.
        curPathLossRecs = pathLossesWithGpsInfo(...
            contains(relPathsOutFilesUnderDataFolder, ...
            [curSiteRelPath,filesep]),:);
        [curNumRowRecords, ~] = size(curPathLossRecs);
        % For updating pathLossesWithGpsAndPlusCoor.
        indicesToWriteAt = counterRowRecord+(1:curNumRowRecords);
        
        % Map the records to the plus shape.
        try
            [xs, zs, hMissingSamps] ...
                = mapPathLossRecsToPlusPatternCoordinates( ...
                relPathsOutFilesUnderDataFolder(...
                counterRowRecord+(1:curNumRowRecords)...
                ));
        catch
            % Not able to assign locations for this site.
            warning(['Skiping site: ',curSiteRelPath, ...
                ' (Unable to assign samples to the plus pattern)'])
            counterRowRecord = counterRowRecord+curNumRowRecords;
            continue;
        end
        
        % Save the result.
        pathLossesWithGpsAndPlusCoor(...
            indicesToWriteAt,8) = xs;
        pathLossesWithGpsAndPlusCoor(...
            indicesToWriteAt,9) = zs;
        
        % Plot.
        hPathLossOnPlusPattern = figure; colormap jet;
        plot3k([xs, pathLossesWithGpsAndPlusCoor(indicesToWriteAt,1), zs], ...
            'ColorData', pathLossesWithGpsAndPlusCoor(indicesToWriteAt,1), ...
            'Marker', {'.', 12});
        grid on; view(0,0); axis tight;
        title(['Path Losses on Plus Pattern - ', date, ' ', type, ...
            ' Series ', num2str(serNum)]);
        xlabel('x (m)'); ylabel('Basic Transmission Loss (dB)');
        zlabel('z (m)');
        
        % Save the plot.
        fullPathToSavePlot = fullfile(ABS_PATH_TO_SAVE_PLOTS, ...
            ['pathLossOnPlusPattern_', strrep(curSiteRelPath, filesep, '_')]);
        saveas(hPathLossOnPlusPattern, ...
            [fullPathToSavePlot, '.fig']);
        saveas(hPathLossOnPlusPattern, ...
            [fullPathToSavePlot, '.png']);
        if isgraphics(hMissingSamps)
            fullPathToSavePlot = [fullPathToSavePlot, '_missingSamples'];
            saveas(hMissingSamps, ...
                [fullPathToSavePlot, '.fig']);
            saveas(hMissingSamps, ...
                [fullPathToSavePlot, '.png']);
        end
        close all;
        
        counterRowRecord = counterRowRecord+curNumRowRecords;
    end
end

% Save results to a .mat file.
fullPathToSaveResults = fullfile(ABS_PATH_TO_SAVE_PLOTS, ...
    'pathLossesWithGpsAndPlusCoor.mat');
save(fullPathToSaveResults, ...
    'pathLossesWithGpsAndPlusCoor');

disp('    Done!')

% EOF