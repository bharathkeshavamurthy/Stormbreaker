% PLOTSERIESLOCATIONSONMAP Plot the GPS points for the series data in a
% specified folder on a Google map.
%
% All data for each series should be organized in its own folder with name
% "Series_#" (e.g. "Series_1"). The GPS data should be contained in files
% named like "measureSignal_1497709963_GPS.log".
%
% Yaguang Zhang, Purdue, 06/18/2017

%% Load data and set the current Matlab directory.

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd('..'); setPath;

PATH_FOLDER_TO_PROCESS = fullfile(pwd, '..', '..', '..', 'Data', ...
    '20170621_MISO'); %'20170617_LargeScale');

%% For each folder, read in all the GPS log files.
dirsToProcess = dir(PATH_FOLDER_TO_PROCESS);
seriesGpsS = cell(length(dirsToProcess),1);
for idxDir = 1:length(dirsToProcess)
    if dirsToProcess(idxDir).isdir
        % Check the folder's name.
        idxSeriesTokens = regexp(dirsToProcess(idxDir).name, ...
            '^Series_(\d+)$', 'tokens');
        if(length(idxSeriesTokens)==1)
            idxSeries = str2double(idxSeriesTokens{1}{1});
            gpsLogs = rdir(fullfile(PATH_FOLDER_TO_PROCESS, ...
                ['Series_', num2str(idxSeries)],'*_GPS.log'));
            % Load the GPS samples.
            seriesGpsS{idxDir} = arrayfun(...
                @(log) parseGpsLog(log.name), gpsLogs);
        end
    end
end
% Remove empty cells.
seriesGpsS = seriesGpsS(~cellfun('isempty',seriesGpsS));

%% Plot each GPS sample on a google map.
seriesColors = colormap(parula);
[numSeriesColors, ~] = size(seriesColors);
indicesColorToUse = randi([1 numSeriesColors],1,length(seriesGpsS));
close all;
hFig = figure; hold on;
markerSize = 10;
for idxSeries = 1:length(seriesGpsS)
    colorToUse = seriesColors(indicesColorToUse(idxSeries),:);
    for idxGpsS = 1:length(seriesGpsS{idxSeries})
        gpggaStr = seriesGpsS{idxSeries}(idxGpsS).gpsLocation;
        gpsLoc = nmealineread(gpggaStr);
        % Add a minus sign if it is W or S.
        if(isW(gpggaStr))
            gpsLoc.longitude = -gpsLoc.longitude;
        end
        if(isS(gpggaStr))
            gpsLoc.latitude = -gpsLoc.latitude;
        end
        % Only plot valid points.
        if (gpsLoc.latitude~=0 && gpsLoc.longitude~=0)
            % Differenciate GPS locked samples and not locked ones.
            if(str2double(seriesGpsS{idxSeries}(idxGpsS).gpsLocked))
                % Locked.
                hLockedNew = plot(gpsLoc.longitude, gpsLoc.latitude, ...
                    '.', 'Color', colorToUse, 'MarkerSize', markerSize);
                if isvalid(hLockedNew)
                    hLocked = hLockedNew;
                end
            else
                % Not locked.
                hUnLockedNew = plot(gpsLoc.longitude, gpsLoc.latitude, ...
                    'x', 'Color', colorToUse, 'MarkerSize', markerSize);
                if isvalid(hUnLockedNew)
                    hUnLocked = hUnLockedNew;
                end
            end
        end
    end
end
plot_google_map('MapType', 'satellite');
hold off;
if exist('hUnLocked')
    legend([hLocked, hUnLocked], 'Locked','Unlocked');
else
    legend([hLocked], 'Locked');
end

%% Save the plot.
pathFolderToSaveFigure = fullfile(PATH_FOLDER_TO_PROCESS, '_results_post_process');
if exist(pathFolderToSaveFigure, 'dir')~=7
    mkdir(pathFolderToSaveFigure);
end
pathFileToSave = fullfile(pathFolderToSaveFigure, 'gpsSamplesOnMap');
saveas(hFig, [pathFileToSave, '.fig']);
saveas(hFig, [pathFileToSave, '.png']);

% EOF