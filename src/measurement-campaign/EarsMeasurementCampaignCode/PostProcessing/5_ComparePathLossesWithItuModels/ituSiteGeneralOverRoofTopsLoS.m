function [ pathLossInDbGaussian ] ...
    = ituSiteGeneralOverRoofTopsLoS( fInGHz, dInM )
%ITUSITEGENERALOVERROOFTOPSLOS To compute the path loss for LoS propagation
%over roof-tops using the site-general ITU model.
%
% This model can be used for environments:
%    - Urban high-rise
%   -  Urban low-rise / Suburban
%
% Ref: ITU-R P.1411-9 (06/2017) Annex 1 Section 4.2.1.
%
% Yaguang Zhang, Purdue, 10/16/2017

%% Parameters

F_IN_GHZ_RANG = [2.2, 73];
D_IN_M_RANGE = [55, 1200];

ALPHA = 2.29;
BETA = 28.6;
GAMMA = 1.96;
SIGMA = 3.48;

% Make sure the inputs are within the required ranges.
if (fInGHz<F_IN_GHZ_RANG(1) || fInGHz>F_IN_GHZ_RANG(2))
    error(['Input fInGHz is out of required range for the ITU model: ', ...
        num2str(F_IN_GHZ_RANG(1)), '~', num2str(F_IN_GHZ_RANG(2))]);
end
if (dInM<D_IN_M_RANGE(1) || dInM>D_IN_M_RANGE(2))
    warning(['Input dInM is out of required range for the ITU model: ', ...
        num2str(D_IN_M_RANGE(1)), '~', num2str(D_IN_M_RANGE(2))]);
end

%% Calculation

% [ pathLossInDbGaussian ] = ituSiteGeneralModel( fInGHz, dInM, ...
%     alpha, beta, gamma, sigma)
[ pathLossInDbGaussian ] = ituSiteGeneralModel( fInGHz, dInM, ...
    ALPHA, BETA, GAMMA, SIGMA);

end
% EOF