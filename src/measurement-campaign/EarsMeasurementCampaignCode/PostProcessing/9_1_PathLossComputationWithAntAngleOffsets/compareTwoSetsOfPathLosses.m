function [ hFigCompPathLosses, hFigCompDifferences ] = compareTwoSetsOfPathLosses( ...
    pathLossesWithGpsInfoOri, pathLossesWithGpsInfoNew)
%COMPARETWOSETSOFPATHLOSSES Compare two different sets of path loss
%computation restuls.
%
% Yaguang Zhang, Purdue, 02/14/2018

minPathLoss = min([pathLossesWithGpsInfoOri(:,1); ...
    pathLossesWithGpsInfoNew(:,1)]);
maxPathLoss = max([pathLossesWithGpsInfoOri(:,1); ...
    pathLossesWithGpsInfoNew(:,1)]);

pathLossDiffs = pathLossesWithGpsInfoNew(:,1)-pathLossesWithGpsInfoOri(:,1);
RMSE = sqrt(mean( ...
    (pathLossDiffs).^2 ...
    ));

% Path losses: new vs original.
hFigCompPathLosses = figure;
hold on;
hComp = scatter(pathLossesWithGpsInfoOri(:,1), ...
    pathLossesWithGpsInfoNew(:,1), ...
    5, ...
    'MarkerFaceColor','b', 'MarkerEdgeColor','none', 'MarkerFaceAlpha', 0.2);
hEqua = plot([minPathLoss, maxPathLoss], [minPathLoss, maxPathLoss], '--k');
h3Db = plot([minPathLoss, maxPathLoss], [minPathLoss, maxPathLoss]+3, ...
    '-.', 'Color', ones(1,3)*0.5);
plot([minPathLoss, maxPathLoss], [minPathLoss, maxPathLoss]-3, ...
    '-.', 'Color', ones(1,3)*0.5);
xlabel('Ori. Path Losses (dB)');
ylabel(['New Path Losses (dB), RMSE = ', num2str(RMSE),' dB']);
legend([hComp, hEqua, h3Db], ...
    'New vs Ori.', 'Ref: Equal', 'Ref: 3dB offsets', ...
    'Location', 'southeast');
grid on; axis equal;

% Path loss difference distribution (relative frequency vs path loss
% difference).
hFigCompDifferences = figure;
hold on;
histogram(pathLossDiffs, 100);
% yticklabels( ...
%     arrayfun(@(tick) {num2str(tick/length(pathLossDiffs))}, yticks)' ...
%      );
xlabel('Path Loss Differences (dB)');
ylabel('# Occurrences');
grid on;
end
% EOF