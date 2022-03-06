function [ pathLossInDbGaussian ] = ituSiteGeneralModel( fInGHz, dInM, ...
    alpha, beta, gamma, sigma)
%ITUSITEGENERALMODEL To compute path loss using the site-general ITU model.
%
% Inputs:
%   - fInGHz
%     The operating frequency in GHz.
%   - dInM
%     3D direct distance between TX and RX.
%   - alpha
%     The coefficient associated with the increase of the path loss with
%     distance.
%   - beta
%     The coefficient associated with the offset value of the path loss.
%   - gamma
%     The coefficient associated with the increase of the path loss with
%     frequency.
%   - sigma
%     The standard deviation in dB for the random offset of the path loss,
%     which is modeled using a zero-mean Gaussian random variable.
%
% Output:
%   - pathLossInDbGaussian
%     A struct for the resulted path loss, which represents a Gaussian
%     random variable with mean and varience specified by the fields
%     pathLossInDbMean and pathLossInDbVar.
%
% Ref: ITU-R P.1411-9 (06/2017) Annex 1 Section 4.1.1 Formula (1).
%
% Yaguang Zhang, Purdue, 10/16/2017

pathLossInDbMean = 10.*alpha.*log10(dInM) + beta + 10.*gamma.*log10(fInGHz);
pathLossInDbVar = sigma;

pathLossInDbGaussian = struct('pathLossInDbMean', pathLossInDbMean, ...
    'pathLossInDbVar', pathLossInDbVar);

end
% EOF