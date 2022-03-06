% MINIMIZERMSEOVERDBPERM Minimize the RMSE for the vegetation path loss by
% varying dBPerM.
%
% Yaguang Zhang, Purdue, 11/13/2017

testRange = -1:0.01:1;

numTests = length(testRange);
RMSEs = nan(numTests,1);
for idxTest = 1:numTests
    disp('========================================')
    disp('    minimizeRmseOverDbPerM progress: ')
    disp(['        ', num2str(idxTest), '/', num2str(numTests)]);
    disp('========================================')
    dBPerM = testRange(idxTest);
    simpleModelsForNLoSByVegetations;
    disp(['         Resulting MSER: ', num2str(rmseAggVegDbPerM)]);
    disp('========================================')
    RMSEs(idxTest) = rmseAggVegDbPerM;
end

[minRMSE, idxMinRMSE] = min(RMSEs);
hMinResult = figure; hold on;
plot(testRange', RMSEs);
plot(testRange(idxMinRMSE), minRMSE, 'rx');
xlabel('Foliage Attenuation (dB/m)'); ylabel('RMSE');
hold off;

saveas(hMinResult, fullfile(pwd, '8_SimpleModelsForNLoSByVegetations', 'minRmse.fig'));
saveas(hMinResult, fullfile(pwd, '8_SimpleModelsForNLoSByVegetations', 'minRmse.png'));
% EOF