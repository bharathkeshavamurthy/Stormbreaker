
c = 30e6;

r = read_complex_binary('r.out',c);
Md = read_float_binary('Md.out',c);
Pd = read_complex_binary('Pd.out',c);
Rd = read_complex_binary('sc_sync0_Rd.out',c);

trv=csvread('trigger')+1;
tr = zeros(length(Md),1);
tr(trv)=25;

Md(Md>2)=2;

L = 64;
Ns = 2*L;
Nc = 12;
F = (Ns+Nc)*128;

%t=1:9e6;
%t = 30.0e6:34e6;%+(1:F*40);
%t=13e6:15e6;
t=1:length(Md)-1000;

%plot(t, [Md(t) 5*tr(t)])

figure(1);
plot(t, [10*log10([abs(Pd(t)) abs(Rd(t)) abs(r(t)).^2]) Md(t) tr(t)])
legend('Pd','Rd','r','Md','tr')
%plot(t, [(20*log10(abs(r(t)))) Md(t) tr(t)])
%legend('r','Md','tr')

%figure(2);plot(t, Md(t))

Rd_c = zeros(length(r)-L,1);
Rd_c(L) = sum(abs(r(1:L)).^2);
for i=2:(length(r)-2*L)
	Rd_c(L+i-1) = sum(abs(r(i:(L+i-1))).^2);
end

figure(2);
plot(t, [10*log10([abs(Rd(t)) Rd_c(t)]) tr(t)])
legend('Rd','Rd (c)','tr')
