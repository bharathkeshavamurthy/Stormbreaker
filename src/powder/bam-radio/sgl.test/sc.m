clear;

c = 10e6;

r = read_complex_binary('r.out',c);
Md = read_float_binary('Md.out',c);
Pd = read_complex_binary('Pd.out',c);
Rd = sqrt(read_float_binary('sc_sync0_Rd2.out',c));

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
t=1e6:1.5e6;
t=1:length(Md)-256;

%plot(t, [Md(t) 5*tr(t)])

figure(1);
plot(t, [10*log10([abs(Pd(t)) Rd(t) abs(r(t)).^2]) Md(t) tr(t)])
legend('Pd','Rd','r','Md','tr')
%plot(t, [(20*log10(abs(r(t)))) Md(t) tr(t)])
%legend('r','Md','tr')

%figure(2);plot(t, Md(t))

Rd_c = zeros(length(r),1);
for i=1:(length(r)-2*L)
	Rd_c(2*L+i-1) = sum(abs(r((i+L):(i+2*L-1))).^2);
end

Pd_c = zeros(length(r),1);
for i=1:(length(r)-2*L)
	Pd_c(2*L+i-1) = sum(r((i):(i+L-1)).*conj(r((i+L):(i+2*L-1))));
end

figure(2);
plot(t, [10*log10([Rd(t) Rd_c(t) abs(Pd(t)) abs(Pd_c(t)) ]) tr(t)./max(tr)*25])
legend('Rd','Rd (c)','Pd','Pd (c)','tr')
figure(3);
plot(t, [10*log10([Rd(t)./Rd_c(t) abs(Pd(t))./abs(Pd_c(t)) ]) tr(t)./max(tr)*1])
legend('Rd d','Pd d','tr')
