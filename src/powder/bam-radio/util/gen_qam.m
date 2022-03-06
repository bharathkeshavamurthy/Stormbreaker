% generate some C code for gray mapped qam constellations
% I do not claim that this is not dirty
function gen_qam(order)
qq = comm.RectangularQAMModulator('ModulationOrder', order, 'SymbolMapping', 'gray');
qq_points = constellation(qq);
qq_E = sum(abs(qq_points).^2)/length(qq_points);
% fudge factor time!
ffs = 'sqrt(2)';
if order == 128
    ffs = '1.7';
end
fprintf('const float level = 1/sqrt(%.2f * %s);\n', qq_E, ffs);
fprintf('this->d_constellation.resize(%d);\n', length(qq_points));
for ii=1:length(qq_points)
    fprintf('this->d_constellation[%d] = gr_complex(%.4f * level, %.4f * level);\n', ...
            ii-1, real(qq_points(ii)), imag(qq_points(ii)));
end
fprintf('\n\nunsigned int bits_per_symbol() const { return %d; }\n\n', log2(order));
figure();
constplot(qq_points * 1/sqrt(qq_E * eval(ffs)));
title(sprintf('%d-QAM', order));
