% VERIFYSIGNALPRESENCEALLFORONESERIES Plot all the signal captured by USRP
% B200 for one series in the EARS measurement campaign.
%
% Yaguang Zhang, Purdue, 06/18/2017

%% Load data and set the current Matlab directory.

% Add libs to current path and set ABS_PATH_TO_EARS_SHARED_FOLDER according
% to the machine name.
cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
cd('..'); setPath;

PATH_FOLDER_TO_PROCESS = fullfile(pwd, '..', '..', '..', 'Data', '20170617_LargeScale');
% Use this to limit what subfolders will be processed.
subfolderPattern = '^Series_([1])$'; % '^Series_(\d+)$';

%% For each folder, read in all the .out files.
dirsToProcess = dir(PATH_FOLDER_TO_PROCESS);
seriesSignalFiltered = cell(length(dirsToProcess),1);
seriesSignal = cell(length(dirsToProcess),1);
for idxDir = 1:length(dirsToProcess)
    if dirsToProcess(idxDir).isdir
        % Check the folder's name.
        idxSeriesTokens = regexp(dirsToProcess(idxDir).name, ...
            subfolderPattern, 'tokens');
        if(length(idxSeriesTokens)==1)
            idxSeries = str2double(idxSeriesTokens{1}{1});
            signalFilteredLogs = rdir(fullfile(PATH_FOLDER_TO_PROCESS, ...
                ['Series_', num2str(idxSeries)],'*_filtered.out'));
            % Load the signal samples.
            seriesSignalFiltered{idxDir} = arrayfun(...
                @(log) read_complex_binary(log.name), ...
                signalFilteredLogs, ...
                'UniformOutput', false);
            seriesSignal{idxDir} = arrayfun(...
                @(log) read_complex_binary(...
                regexprep(log.name, '_filtered','')),...
                signalFilteredLogs, ...
                'UniformOutput', false);
        end
    end
end
% Remove empty cells.
seriesSignalFiltered = seriesSignalFiltered(...
    ~cellfun('isempty',seriesSignalFiltered));
seriesSignal = seriesSignal(~cellfun('isempty',seriesSignal));

%% Plot the signals.

% We will try to find the "tallest" bump.
numPreSamples = 200;
numPostSamples = 2000;
MAX_NUM_FIGS = 10;
for idxSeries = 1:length(seriesSignal)
    numFigs = 0;
    for idxSignalFiles = 1:length(seriesSignal{idxSeries})
        figureSupTitle = ['Series ', num2str(idxSeries), ...
            ' - ', num2str(idxSignalFiles)];
        plotOnePresentSignal(...
            seriesSignalFiltered{idxSeries}{idxSignalFiles}, ...
            numPreSamples, numPostSamples, [figureSupTitle, '(Filtered)']);
        numFigs = numFigs+1;
        plotOnePresentSignal(...
            seriesSignal{idxSeries}{idxSignalFiles}, ...
            numPreSamples, numPostSamples, [figureSupTitle]);
        numFigs = numFigs+1;
        if numFigs>=MAX_NUM_FIGS
            pause;
            close all;
            numFigs = 0;
        end
    end
end

% EOF