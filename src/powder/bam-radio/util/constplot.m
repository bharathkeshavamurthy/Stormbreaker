function constplot(data)
x = real(data);
y = imag(data);
scatter(x, y, 'o', 'yellow', 'fill');
ax = gca();
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
ax.Box = 'off';
ax.Layer = 'top';
set(ax,'Color','k');
set(ax,'XColor','white');
set(ax,'Ycolor','white');
title('Constellation')
xlim([min(x) max(x)] + [-1 1] * 0.2);
ylim([min(y) max(y)] + [-1 1] * 0.2);