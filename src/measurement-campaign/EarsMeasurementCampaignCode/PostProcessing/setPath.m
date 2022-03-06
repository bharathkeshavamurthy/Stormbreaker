% SETPATH Add lib folders into Matlab path.
%
% Update 20170915: we will also set ABS_PATH_TO_EARS_SHARED_FOLDER
% according to the current machine's name to make the programs run on
% different known machines without manually modifying the code.
%
% Yaguang Zhang, Purdue, 07/11/2017

cd(fileparts(mfilename('fullpath')));
addpath(fullfile(pwd));
addpath(genpath(fullfile(pwd, 'lib')));

% The absolute path to the shared Google Drive folder "Annapolis
% Measurement Campaign". Please make sure it is correct for the machine
% which will run this script.
%  - On Mac Lemma:
%    '/Users/zhan1472/Google Drive/Annapolis Measurement Campaign';
%  - On Windows Dell to remotely access Lemma:
%    '\\LEMMA\Google Drive\Annapolis Measurement Campaign';
%  - Local copy on Windows Dell:
%    'C:\Users\Zyglabs\Documents\MEGAsync\EARS';
unknownComputerErrorMsg = ...
    ['Compute not recognized... \n', ...
    '    Please update setPath.m for your machine. '];
unknownComputerErrorId = 'setPath:computerNotKnown';
switch getenv('computername')
    case 'ZYGLABS-DELL'
        % ZYG's Dell laptop.
        ABS_PATH_TO_EARS_SHARED_FOLDER = ...
            'C:\Users\Zyglabs\OneDrive - purdue.edu\EARS';
    case 'ARTSY'
        % ZYG's lab desktop.
        ABS_PATH_TO_EARS_SHARED_FOLDER = ...
            'D:\Google Drive - EARS\Annapolis Measurement Campaign';
    case ''
        % Expected to be Lemma the Mac machine in ZYG's lab.
        assert(ismac, unknownComputerErrorMsg);
        ABS_PATH_TO_EARS_SHARED_FOLDER = ...
            '/Users/zhan1472/Google Drive/Annapolis Measurement Campaign';
    otherwise
        error(unknownComputerErrorId, unknownComputerErrorMsg);
end

% EOF