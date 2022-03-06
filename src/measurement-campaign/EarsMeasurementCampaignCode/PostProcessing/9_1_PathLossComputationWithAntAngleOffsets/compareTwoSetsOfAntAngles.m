function [ hFig ] = compareTwoSetsOfAntAngles( ...
    txInfoLogsOri, txInfoLogsNew )
%COMPARETWOSETSOFANTANGLES Compare two different sets of antenna angles
%embedded in the input txInfoLog variables.
%
% Yaguang Zhang, Purdue, 02/14/2018

hFig = figure;
hold on;
for idxDay = 1:length(txInfoLogsOri)
    for idxSite = 1:length(txInfoLogsOri{idxDay})
        
        % If the abs function changed the Azimuth angle, it may happen that
        % the angle difference is larger than 180. We will fix that
        % situation here for proper plotting.
        txAzDiff = fixAzDiffForPlotting( ...
            txInfoLogsNew{idxDay}(idxSite).txAz ...
            - txInfoLogsOri{idxDay}(idxSite).txAz ...
            );
        rxAzDiff = fixAzDiffForPlotting( ...
            txInfoLogsNew{idxDay}(idxSite).rxAz ...
            - txInfoLogsOri{idxDay}(idxSite).rxAz ...
            );
        if any(isnan([txAzDiff rxAzDiff]))
            disp(' ')
            disp('Warning: site with invalid angle difference: ')
            disp(' ')
            disp(txInfoLogsNew{idxDay}(idxSite))
            disp(' ')
        end
        
        hTx = plot(txAzDiff, ...
            txInfoLogsNew{idxDay}(idxSite).txEl ...
            - txInfoLogsOri{idxDay}(idxSite).txEl, 'ob');
        % RX
        hRx = plot(rxAzDiff, ...
            txInfoLogsNew{idxDay}(idxSite).rxEl ...
            - txInfoLogsOri{idxDay}(idxSite).rxEl, 'xr');
    end
end
legend([hTx, hRx], 'For Tx', 'For Rx');
xlabel('Angle Offset - Azimuth');
ylabel('Angle Offset - Elevation');
axis equal;
grid on;

end
% EOF