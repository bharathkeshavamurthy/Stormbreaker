function [ alpha, beta, fittedFct ] ...
    = fitAlphaBetaWithGivenGamma(ds, lossesInDb, fInGHz, gamma)
%FITALPHABETAWITHGIVENGAMMA Fit the input distances and the cooresponding
%path losses with the Alpha/Beta/Gamma model model with a custom fixed
%gamma for a specified frequence.
%
% The Alpha/Beta/Gamma model we use:
%   PL(d) = 10*alpha*log10(d/d0) + beta + 10*gamma*log10(frequency in GHz)
%
% Outputs alpha and beta are the fitted parameters; And fittedFct is the
% resulted function with alpha and beta used for the model.
%
% Yaguang Zhang, Purdue, 10/19/2017

if ~isequal(size(ds), size(lossesInDb))
    error('fitCloseIn:InputsSizeMismatch', ...
        'ds and lossesInDb vectors must be of the same size.');
end

% 1 m close-in reference point.
d0 = 1;
% Fitting.
AlphaBetaWithGivenGammaModel ...
    = @(paras, d) 10.*paras(1).*log10(d./d0) + paras(2) ...
    + 10.*gamma.*log10(fInGHz);
% Use random initial values.
results = nlinfit(ds, lossesInDb, AlphaBetaWithGivenGammaModel, rand(1,2));

alpha = results(1);
beta = results(2);
fittedFct = @(d) 10.*alpha.*log10(d./d0) + beta ...
    + 10.*gamma.*log10(fInGHz);
end
% EOF