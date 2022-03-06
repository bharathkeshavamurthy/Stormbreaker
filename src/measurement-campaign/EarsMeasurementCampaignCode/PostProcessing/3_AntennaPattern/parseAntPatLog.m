function [ patAtIntFreq, fileHeaders ] = parseAntPatLog( absFilePath, ...
    scanndedFreqs, intFreq )
%PARSEANTENNAPATTERNLOG Load and parse an antenna pattern .txt log file.
%
% We expect the file to have sample lines to have a structure (possibly
% with other lines which do not agree with this):
%   Azimuth   Elevation   [ |S21| phase(S21) ]*numberOfFrequenciesScanned
% where each scanned frequency will add a corresponding [|S21| phase(S21)]
% part (which contains two new columns). And it doesn't matter how long the
% white spaces are between columns.
%
% Inputs:
%   - absFilePath
%     A string for the absolute path to the antenna pattern log file.
%   - scanndedFreqs
%     A vector specifying all the frequencies scanned. Essentially, this
%     will help us find which columns of data to load.
%   - intFreq
%     The intermediate / scanned frequency we care about.
%
% Outputs:
%   - patAtIntFreq
%     The antenna pattern at intFreq, which is a struct containing fields:
%       - azs
%         The azimuth angles in degree.
%       - els
%         The elevation angles in degree.
%       - amps
%         The linear amplitudes of the samples.
%       - phases
%         The phases of the samples.
%     All of these fields contains a column vector with each row
%     corresponding to a sweep sample.
%   - fileHeaders
%     Optional cell array output which contains the string header lines.
%
% Yaguang Zhang, Purdue, 06/29/2017

% Text scan pattern. We have columns Azimuth, Elevation, and for every
% scaned frequency, one for Amplitude and another for Phase.
samplePattern = repmat('%f', [1, 2+2.*length(scanndedFreqs)]);

fileID = fopen(absFilePath, 'r');

% The number of header lines for each log file is 2.
fileHeaders = cell(2,1);
% Samples info: Number of azimuths, number of elevations, and number of
% frequences swept.
fileHeaders{1} = fgetl(fileID); 
% Table headers: Azimuth, elevation, and frequencies.
fileHeaders{2} = fgetl(fileID);

% Fetch the samples.
data = textscan(fileID, samplePattern, 'CollectOutput',1);
data = data{1};
fclose(fileID);

% Get the line/sample number and scanned frequency number recorded in the
% header line.
recNums = textscan(fileHeaders{1}, '"%f""%f""%f"');
% numAzs = recNums{1};
% numEls = recNums{2};
numFreqs = recNums{3}; 
assert(length(scanndedFreqs) == numFreqs, ...
    ['The scanned frequencies should have ', num2str(numFreqs), ' points!']);

% Only keep the data for the frequency we need.
patAtIntFreq = struct('azs', [], 'els', [], 'amps', [], 'phases', []);
patAtIntFreq.azs = data(:,1);
patAtIntFreq.els = data(:,2);
idxAmpCol = 1 + 2.*find(scanndedFreqs == intFreq, 1);
patAtIntFreq.amps = data(:,idxAmpCol);
patAtIntFreq.phases = data(:,idxAmpCol+1);

end
% EOF