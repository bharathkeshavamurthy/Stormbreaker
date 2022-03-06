% GENPLOTS Generate plots for the summary report and save them as .png
% figures.
%
% We will process all the data collected and duplicate the folder structure
% for them into a "PostProcessingResults/SummaryReport/plots" folder under
% the shared Google Drive folder.
%
% Yaguang Zhang, Purdue, 07/11/2017

clear; clc; close all;

%% Configurations

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd('..'); setPath;

% Flags to enable coresponding plot functions.
FLAG_PLOT_GPS_FOR_EACH_DAY = true;
FLAG_PLOT_FIRST_SEVERAL_MEAS_PER_SERIES = false;

% Configure other paths accordingly.
ABS_PATH_TO_DATA = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, 'Data');
ABS_PATH_TO_SAVE_PLOTS = fullfile(ABS_PATH_TO_EARS_SHARED_FOLDER, ...
    'PostProcessingResults', 'SummaryReport', 'plots');

% The number of measurements that are considered for each site, i.e. there
% will be numMeasToPlotPerSeries plots generated for the signals located in
% one "Series_xx" folder.
numMeasToPlotPerSeries = 3;

%% Before Generating the Plots

disp(' ---------- ')
disp('  genPlots')
disp(' ---------- ')

% Disable the tex interpreter in figures.
set(0,'DefaultTextInterpreter','none');

% Create directories if necessary.
if exist(ABS_PATH_TO_SAVE_PLOTS, 'dir')~=7
    mkdir(ABS_PATH_TO_SAVE_PLOTS);
end

% Find all the parent directories for "Series_xx" data folders using regex.
disp(' ')

% Try loading the information for the samples first.
pathToPlotInfo = fullfile(ABS_PATH_TO_SAVE_PLOTS, 'plotInfo.mat');
if exist(pathToPlotInfo, 'file')
    % The data have been processed before and the plotInfo.mat file has
    % been found. Load that to save time.
    disp('    Found plotInfo.Mat; Loading the sample info in it...')
    load(pathToPlotInfo);
else
    disp('    No plotInfo.mat found; Searching for "Series" data folders...')
    % Need to actually scan the folder and find the sample folders.
    allSeriesParentDirs = rdir(fullfile(ABS_PATH_TO_DATA, '**', '*'), ...
        'regexp(name, ''(_LargeScale$)|(_SIMO$)|(_Conti$)'')');
    % Also locate all the "Series_xx" data folders for each parent directory.
    allSeriesDirs = cell(length(allSeriesParentDirs),1);
    for idxPar = 1:length(allSeriesParentDirs)
        assert(allSeriesParentDirs(idxPar).isdir, ...
            ['#', num2str(idxPar), ' series parent dir should be a folder!']);
        
        curSeriesDirs = rdir(fullfile(allSeriesParentDirs(idxPar).name, '**', '*'), ...
            'regexp(name, ''(Series_\d+$)'')');
        if(isempty(curSeriesDirs))
            warning(['#', num2str(idxPar), ...
                ' series parent dir does not have any series subfolders!']);
        end
        allSeriesDirs{idxPar} = curSeriesDirs;
    end
    disp('    Saving the results...')
    % Note that the exact paths to the folders may change depending on the
    % manchine and its operating system, so only the folder names should be
    % used.
    save(pathToPlotInfo, ...
        'allSeriesParentDirs', 'allSeriesDirs');
end
disp('    Done!')

%% Google Maps for Each Parent Folder

if FLAG_PLOT_GPS_FOR_EACH_DAY
    TX_LAT = 38.983899;
    TX_LON = -76.486682;
    
    disp(' ')
    disp('  => Plotting GPS samples on Google map...')
    for idxPar = 1:length(allSeriesParentDirs)
        disp([num2str(idxPar), '/', num2str(length(allSeriesParentDirs))])
        PATH_FOLDER_TO_PROCESS = allSeriesParentDirs(idxPar).name;
        
        % For each folder, read in all the GPS log files.
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
        
        % Plot each GPS sample on a google map.
        seriesColors = colormap(parula);
        [numSeriesColors, ~] = size(seriesColors);
        indicesColorToUse = randi([1 numSeriesColors],1,length(seriesGpsS));
        close all;
        hFigGpsOnMap = figure; hold on;
        markerSize = 10;
        % Plot Tx.
        hTx = plot(TX_LON, TX_LAT, 'r*', 'MarkerSize', markerSize);
        for idxSeries = 1:length(seriesGpsS)
            % Keep a record of the locked samples.
            lockedLats = [];
            lockedLons = [];
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
                        lockedLats(end+1) = gpsLoc.latitude;
                        lockedLons(end+1) = gpsLoc.longitude;
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
            if ~isempty(lockedLats)
                % Plot the distance from the GPS cluster center (median
                % locked lon, median locked lat) to the Tx.
                latMedian = median(lockedLats);
                lonMedian = median(lockedLons);
                
                % Link the cluster center with the Tx.
                hDistLine = plot([lonMedian, TX_LON], ...
                    [latMedian, TX_LAT], 'r-', 'LineWidth', 0.5);
                hClusterCenter = plot(lonMedian, latMedian, 'r.');
                % Move the distance line indicator to the bottom.
                uistack(hDistLine, 'bottom');
                
                % Calculate the distance in meters.
                distToTx = 1000* ...
                    lldistkm([latMedian, lonMedian],[TX_LAT, TX_LON]);
                % Label it on the plot, too.
                text(lonMedian, latMedian, ...
                    [num2str(distToTx, '%.1f'), ' m'], 'Color', 'y', ...
                    'VerticalAlignment', 'top');
            end
        end
        plot_google_map('MapType', 'satellite');
        hold off;
        if exist('hUnLocked', 'var')
            legend([hTx, hLocked, hUnLocked], 'Tx', 'Locked','Unlocked');
        elseif exist('hLocked', 'var')
            legend([hTx, hLocked], 'Tx', 'Locked');
        end
        clear hLocked hUnLocked;
        [~, seriesParentDirName] = fileparts(allSeriesParentDirs(idxPar).name);
        title(seriesParentDirName, 'Interpreter', 'none');
        
        % Save the plot.
        pathFileToSave = fullfile(ABS_PATH_TO_SAVE_PLOTS, ...
            [seriesParentDirName, '_gpsSamplesOnMap']);
        saveas(hFigGpsOnMap, [pathFileToSave, '.fig']);
        saveas(hFigGpsOnMap, [pathFileToSave, '.png']);
    end
    disp('     Done!')
end

%% Verify Signal Present for the First numMeasToPlotPerSeries Measurements

if FLAG_PLOT_FIRST_SEVERAL_MEAS_PER_SERIES
    disp(' ')
    disp('  => Plotting the first estimated signal for a few measurement on each site...')
    for idxPar = 1:length(allSeriesParentDirs)
        disp(['Loading data for parent dir ', num2str(idxPar), '/', ...
            num2str(length(allSeriesParentDirs)), '...'])
        PATH_FOLDER_TO_PROCESS = allSeriesParentDirs(idxPar).name;
        % Use this to limit what subfolders will be processed.
        subfolderPattern = '^Series_(\d+)$';
        
        % For each folder, read in all the .out files needed. To save
        % memory, we will generate plots for each series, one by one.
        dirsToProcess = dir(PATH_FOLDER_TO_PROCESS);
        numSeriesInParDir = length(rdir(fullfile(PATH_FOLDER_TO_PROCESS, '*'), ...
            'regexp(name, ''(Series_\d+$)'')'));
        for idxDir = 1:length(dirsToProcess)
            if dirsToProcess(idxDir).isdir
                % Check the folder's name.
                idxSeriesTokens = regexp(dirsToProcess(idxDir).name, ...
                    subfolderPattern, 'tokens');
                if(length(idxSeriesTokens)==1)
                    idxSeries = str2double(idxSeriesTokens{1}{1});
                    signalFilteredLogs = rdir(fullfile(PATH_FOLDER_TO_PROCESS, ...
                        ['Series_', num2str(idxSeries)],'*_filtered.out'));
                    % Ignore measurements that will not be used.
                    signalFilteredLogs = signalFilteredLogs(1:min( ...
                        [numMeasToPlotPerSeries; length(signalFilteredLogs)]));
                    % Load the signal samples. The integer countSam is to
                    % limit the number samples to load.
                    countSam = 1.04*(10^6); % ~ 1s of the signal.
                    seriesSignalFiltered = arrayfun(...
                        @(log) read_complex_binary(log.name, countSam), ...
                        signalFilteredLogs, ...
                        'UniformOutput', false);
                    seriesSignal = arrayfun(...
                        @(log) read_complex_binary(...
                        regexprep(log.name, '_filtered',''), countSam),...
                        signalFilteredLogs, ...
                        'UniformOutput', false);
                    if (length(seriesSignal)<numMeasToPlotPerSeries)
                        warning(['#', num2str(idxPar), ...
                            ' series parent folder does not have enough valid measuremnts loaded!']);
                    end
                    if ~isempty(seriesSignalFiltered)
                        % Plot the signals. We will try to find the
                        % "tallest" bump for each measurement.
                        numPreSamples = 200;
                        numPostSamples = 2000;
                        
                        disp(['Generating plots for series ', num2str(idxSeries), '/', ...
                            num2str(numSeriesInParDir), '...'])
                        for idxSignalFiles = 1:length(seriesSignal)
                            close all;
                            [~, seriesParentDirName] = fileparts(allSeriesParentDirs(idxPar).name);
                            figureSupTitle = [seriesParentDirName, ': Series ', num2str(idxSeries), ...
                                ' - ', num2str(idxSignalFiles)];
                            hFigSigFiltered = plotOnePresentSignal(...
                                seriesSignalFiltered{idxSignalFiles}, ...
                                numPreSamples, numPostSamples, [figureSupTitle, ' (Filtered)']);
                            hFigSig = plotOnePresentSignal(...
                                seriesSignal{idxSignalFiles}, ...
                                numPreSamples, numPostSamples, figureSupTitle);
                            % Save the plots.
                            pathFileToSave = fullfile(ABS_PATH_TO_SAVE_PLOTS, ...
                                [seriesParentDirName, '_oneSigPerMeas_series_', ...
                                num2str(idxSeries), '_meas_', num2str(idxSignalFiles)]);
                            saveas(hFigSigFiltered, [pathFileToSave, '_filtered.fig']);
                            saveas(hFigSigFiltered, [pathFileToSave, '_filtered.png']);
                            saveas(hFigSig, [pathFileToSave, '.fig']);
                            saveas(hFigSig, [pathFileToSave, '.png']);
                        end
                    end
                end
            end
        end
    end
    disp('     Done!')
end
% EOF