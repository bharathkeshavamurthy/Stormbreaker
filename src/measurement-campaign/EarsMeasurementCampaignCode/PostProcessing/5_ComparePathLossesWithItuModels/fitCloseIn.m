function [ n, fittedFct ] = fitCloseIn(ds, lossesInDb, fInHz)
%FITCLOSEIN Fit the input distances and the cooresponding path losses with
%the close-in model for a specified frequence.
%
% The close-in model we use:
%   PL(d) = PL(d0) + 10*n*log10(d/d0)
% where PL(d0) is the free-space propagation loss at a distance of d0=1m
% with isotropic antennas.
% 
% Output n is the fitted parameter n and fittedFct is the resulted function
% with n used for the model.
% 
% Yaguang Zhang, Purdue, 10/19/2017

if ~isequal(size(ds), size(lossesInDb))
    error('fitCloseIn:InputsSizeMismatch',...
          'ds and lossesInDb vectors must be of the same size.');
end

% Wavelength.
lambda = physconst('LightSpeed')/fInHz;

% 1 m close-in reference point.
d0 = 1;
Ld0 = fspl(d0, lambda);

% Fitting.
closeInModel = @(nToFit,d) Ld0 + 10.*nToFit.*log10(d./d0);
n0 = 2;
n = nlinfit(ds, lossesInDb, closeInModel, n0);
fittedFct = @(d) Ld0 + 10.*n.*log10(d./d0);
end
% EOF