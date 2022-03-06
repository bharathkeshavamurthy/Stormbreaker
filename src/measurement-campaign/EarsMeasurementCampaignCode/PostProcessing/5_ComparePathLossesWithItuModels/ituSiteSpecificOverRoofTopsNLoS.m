function [ LNLoS1 ] ...
    = ituSiteSpecificOverRoofTopsNLoS( fInGHz, dInM, ...
    hRInM, wInM, ... % bInM,
    phiInDegree, h1InM, h2InM, ... %, lInM
    FLAG_IGNORE_OUT_OF_RANGE )
%ITUSITESPECIFICOVERROOFTOPSNLOS To compute the path loss for nLoS
%propagation over roof-tops using the site-specific ITU model (NLoS1).
%
% Inputs:
%   - fInGHz
%     The operating frequency in GHz.
%   - dInM
%     3D direct distance between TX and RX.
%   - hRInM
%     The average height of buildings.
%   - wInM
%     The street width.
%   - bInM (Update: not used in this model)
%     The average building separation.
%   - phiInDegree
%     The street orientation with respect to the direct path.
%   - h1InM
%     The station 1 antenna height.
%   - h2InM
%     The station 2 antenna height.
%   - lInM (Update: not used in this model)
%     The length of the path covered by buildings.
%   - FLAG_IGNORE_OUT_OF_RANGE
%     Set this to be true to force the algorithm to use the inputs as they
%     are.
%
% Output:
%   - LNLoS1
%     The resulted path loss in dB.
%
% Test:
%       ituSiteSpecificOverRoofTopsNLoS( 28, 500, ...
%           13, 28, ...
%               30, 27, 1.5)
%
% Ref: ITU-R P.1411-9 (06/2017) Annex 1 Section 4.2.2.
%
% Yaguang Zhang, Purdue, 10/16/2017

%% Parameters

if nargin < 8
    FLAG_IGNORE_OUT_OF_RANGE = false;
end

F_IN_GHZ_RANG = [0.8, 38];
D_IN_M_RANGE = [10, 5000];

W_IN_M_RANGE = [10, 25];

DELTA_H_1_IN_M_RANGE = [1, 100];
DELTA_H_2_IN_M_RANGE = [4, min(10, hRInM)];

% Make sure the inputs are within the required ranges.
deltaH1InM = h1InM - hRInM;
deltaH2InM = hRInM - h2InM;
if ~FLAG_IGNORE_OUT_OF_RANGE
    if (fInGHz<F_IN_GHZ_RANG(1) || fInGHz>F_IN_GHZ_RANG(2))
        error(['Input fInGHz is out of required range for the ITU model: ', ...
            num2str(F_IN_GHZ_RANG(1)), '~', num2str(F_IN_GHZ_RANG(2))]);
    end
    if (dInM<D_IN_M_RANGE(1) || dInM>D_IN_M_RANGE(2))
        error(['Input dInM (', num2str(dInM), ...
            ') is out of required range for the ITU model: ', ...
            num2str(D_IN_M_RANGE(1)), '~', num2str(D_IN_M_RANGE(2))]);
    end
    if (wInM<W_IN_M_RANGE(1) || wInM>W_IN_M_RANGE(2))
        warning(['Input wInM is out of required range for the ITU model: ', ...
            num2str(W_IN_M_RANGE(1)), '~', num2str(W_IN_M_RANGE(2))]);
        if wInM<W_IN_M_RANGE(1)
            wInM = W_IN_M_RANGE(1);
            warning('dInM has been set to the LOWER bound!');
        else % dInM>D_IN_M_RANGE(2)
            wInM = W_IN_M_RANGE(2);
            warning('dInM has been set to the UPPER bound!');
        end
    end
    if (deltaH1InM<DELTA_H_1_IN_M_RANGE(1) ...
            || deltaH1InM>DELTA_H_1_IN_M_RANGE(2))
        error(['Resulted deltaH1InM is out of required range for the ITU model: ', ...
            num2str(DELTA_H_1_IN_M_RANGE(1)), '~', ...
            num2str(DELTA_H_1_IN_M_RANGE(2))]);
    end
    if (deltaH2InM<DELTA_H_2_IN_M_RANGE(1) ...
            || deltaH2InM>DELTA_H_2_IN_M_RANGE(2))
        warning(['Resulted deltaH2InM ', num2str(deltaH2InM), ...
            ' is out of required range for the ITU model: ', ...
            num2str(DELTA_H_2_IN_M_RANGE(1)), '~', ...
            num2str(DELTA_H_2_IN_M_RANGE(2))]);
        %     if deltaH2InM<DELTA_H_2_IN_M_RANGE(1)
        %         deltaH2InM = DELTA_H_2_IN_M_RANGE(1); warning('hRInM has
        %         been set to agree with the deltaH2InM LOWER bound!');
        %     else % dInM>D_IN_M_RANGE(2)
        %         deltaH2InM = DELTA_H_2_IN_M_RANGE(2); warning('hRInM has
        %         been set to agree with the deltaH2InM UPPER bound!');
        %     end hRInM = h2InM + deltaH2InM;
    end
end
%% Calculation

% Wavelength.
lambdaInM = physconst('LightSpeed')./(fInGHz.*(10.^9));

% We need to find k s.t. d_k <= d <= d_(k+1). For simplicity, just compute
% a bunch of d_k's for comparison.
NUM_KS_TO_EVA = 1000;
ks = 0:(NUM_KS_TO_EVA-1);

% Formula (55).
Aks = wInM.*(h1InM - h2InM).*(2.*ks + 1)./(2.*(hRInM - h2InM));
% Formula (56).
Bks = Aks - ks.*wInM;
% Formula (57).
phiks = atand((Aks./Bks).*tand(phiInDegree));

% Formula (54).
dkps = sqrt( (Aks./sind(phiks)).^(2) + (h1InM - h2InM).^(2) );
% Formula (50).
dks = sqrt( (Bks./sind(phiInDegree)).^(2) + (h1InM - h2InM).^(2) );

% Formula (51).
Ldks = 20.*log10( (4.*pi.*dkps) ./ ( (0.4).^(ks) .* lambdaInM ) );
LdksPos = Ldks(2:end);

% Formula (52). Note that the first element in dks is actually d0.
d0 = dks(1);

% Formula (48). Line 1.
if dInM<d0
    LNLoS1 = 20.*log10(4.*pi.*dInM./lambdaInM);
else
    % Formula (52).
    dksPos = dks(2:end);
    dRD = ( 0.25.*dksPos(3) + 0.25.*dksPos(4) ...
        - 0.16.*dksPos(1) - 0.35.*dksPos(2) ) ...
        .*log10(fInGHz) ...
        + 0.25.*dksPos(1) + 0.56.*dksPos(2) ...
        + 0.10.*dksPos(3) + 0.10.*dksPos(4);
    
    % Formula (53). First we need to find k s.t. d_k <= d_RD <= d_(k+1).
    % Note that here we use d_k <= d_RD < d_(k+1) to eliminate ambiguity.
    kLdRd = find(dRD<dks, 1, 'first')-2;
    % To make sure k=0 still works.
    kLdRdPlusOne = kLdRd+1;
    LdRD = Ldks(kLdRdPlusOne) ...
        + (Ldks(kLdRdPlusOne+1)-Ldks(kLdRdPlusOne))...
        ./(dks(kLdRdPlusOne+1)-dks(kLdRdPlusOne)) ...
        .*(dRD-dks(kLdRdPlusOne));
    
    if (d0<=dInM && dInM<dRD)
        % Formula (48). Line 2.
        
        % Formula (49). First we need to find k s.t. d_k <= dInM < d_(k+1).
        k = find(dInM<dks, 1, 'first')-2;
        % To make sure k=0 still works.
        kPlusOne = k+1;
        if (dks(kPlusOne) <= dInM) ...
                && (dInM  <  dks(kPlusOne+1)) ...
                && (dks(kPlusOne+1) < dRD)
            % Formula (49). Line 1.
            LNLoS1 = Ldks(kPlusOne) ...
                + ( Ldks(kPlusOne+1) - Ldks(kPlusOne) ) ...
                ./( dks(kPlusOne+1) - dks(kPlusOne) ) ...
                .*( dInM - dks(kPlusOne) );
        elseif (dks(kPlusOne) <= dInM) ...
                && (dInM  <  dRD) ...
                && (dRD   < dks(kPlusOne+1))
            if (k~=kLdRd)
                warning('The value k for formula (49) and the one for formula (53) do not agree with each other!')
            end
            % Formula (49). Line 2.
            LNLoS1 = Ldks(kPlusOne) ...
                + ( LdRD - Ldks(kPlusOne) ) ...
                ./( dRD - dks(kPlusOne) ) ...
                .*( dInM - dks(kPlusOne) );
        else
            print(['dk=', num2str(dksPos(k)), ' dInM=', num2str(dInM), ...
                ' d(k+1)=', num2str(dksPos(k+1)), ' dRD=', num2str(dRD)]);
            error('Unexpected case for dInM in formula (49)!');
        end
        
    elseif (dInM>=dRD)
        % Formula (48). Line 3.
        LNLoS1 = (32.1).*log10(dInM./dRD) + LdRD;
    else
        print(['d0=', num2str(d0), ' dInM=', num2str(dInM), ...
            ' dRD=', num2str(dRD)]);
        error('Unexpected case for dInM in formula (48)!');
    end
end

end
% EOF