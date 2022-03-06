% Similarly, measured power vs calculated power.
hFigCalibrationMeasVsCalc = figure; hold on;
axisToSet = [ min(calPs) max(calPs) ...
    min(meaPs) max(meaPs)];
hScatterPts = cell(2,1);
% Calibration points.
for idxDataset = 1:numDatasets
    xs = calculatedPowers{idxDataset};
    ys = measPowers{idxDataset};
    
    % For plotting, only show points to fit.
    boolsPtsToFit = BOOLS_MEAS_TO_FIT{idxDataset}';
    xsToShow = xs((~isinf(ys))&boolsPtsToFit);
    ysToShow = ys((~isinf(ys))&boolsPtsToFit);
    
    % Plot the results.
    colorToUse = seriesColors(indicesColorToUse(idxDataset),:);
    % Non-inf points.
    hScatterPts{idxDataset} = scatter(xsToShow, ysToShow, '*', 'MarkerEdgeColor', colorToUse, ...
        'LineWidth',1.5);
end
% Set the visible area of the plot now according to the data points shown.
axis(axisToSet); axis equal; finalAxis = axis; axis manual;
% Add the lslines.
for idxDataset = 1:numDatasets
    xs = calculatedPowers{idxDataset};
    ys = measPowers{idxDataset};
    
    % For fitting, remove points with too low estimated SNR.
    boolsPtsToFit = estimatedSnrs{idxDataset}>=minValidEstSnr;
    % Also get rid of measurements to use in line fitting.
    boolsPtsToFit = boolsPtsToFit & BOOLS_MEAS_TO_FIT{idxDataset}';
    xsToFit = xs(boolsPtsToFit);
    ysToFit = ys(boolsPtsToFit);
    
    % For fitting, remove points with too low calculated power.
    xsToFit = xsToFit(xsToFit>=minValidCalPower);
    ysToFit = ysToFit(ysToFit>=minValidCalPower);
    
% % Cover the unused points in the plot.
%  hIgnoredPts = plot(xs(~boolsPtsToFit),ys(~boolsPtsToFit), ...
%     'r*', 'LineWidth',1.5);
    
    xRangeToShow = linspace(finalAxis(1),finalAxis(2));
    % Plot the fitted line.
    lsLinePolyInv = lsLinesPolysInv{idxDataset};
    valuesLsLine = polyval(lsLinePolyInv, xRangeToShow);
    hLsLines{idxDataset} = ...
        plot(xRangeToShow,valuesLsLine, ...
        'Color',colorToUse,'LineStyle', '--');
    
    % Show the polynomial on the plot.
    if lsLinePolyInv(2)>0
        strPoly=['y = ',num2str(lsLinePolyInv(1)),'x+',num2str(lsLinePolyInv(2))];
    elseif lsLinePolyInv(2)<0
        strPoly=['y = ',num2str(lsLinePolyInv(1)),'x',num2str(lsLinePolyInv(2))];
    else % lsLinePolyInv(2)==0
        strPoly=['y = ',num2str(lsLinePolyInv(1)),'x'];
    end
    idxMiddlePtToFit = floor(length(xsToFit)/2);
    % Black bold copy as background for clarity.
    text(xsToFit(idxMiddlePtToFit), ...
        ysToFit(idxMiddlePtToFit), strPoly, ...
        'Rotation', rad2deg(atan(lsLinePolyInv(1))), ...
        'FontWeight', 'bold', 'Color', 'white', ...
        'VerticalAlignment', 'top');
    text(xsToFit(idxMiddlePtToFit), ...
        ysToFit(idxMiddlePtToFit), strPoly, ...
        'Rotation',rad2deg(atan(lsLinePolyInv(1))), ...
        'Color', seriesColors(indicesColorToUse(idxDataset),:), ...
        'VerticalAlignment', 'top');
end
if ~isinf(minValidCalPower)
    plot([finalAxis(1) finalAxis(2)], [minValidCalPower minValidCalPower], ...
        'r--');
    x = [finalAxis(1) finalAxis(1) finalAxis(2) finalAxis(2)];
    minY = -200;
    y = [minY minValidCalPower minValidCalPower minY];
    patch(x,y,[1,1,1].*0.6,'FaceAlpha',0.3,'LineStyle','none');
end
title('Calibration results');
xlabel('Calculated Power (dB)');
ylabel('Measured Power (dB)'); 
legend([hScatterPts{1} hScatterPts{2}],'1 dB RX gain','76 dB RX gain');
grid on; hold off;