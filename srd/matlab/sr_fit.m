function [fitline,noiseline,results,gof]=sr_fit(f,p_in,modes)

f_res=[7.8, 14, 20, 27, 33, 39, 45];
f_res=f_res(1:modes);

% results [f p q]
[results,BN]=find_resonances(f,p_in);

% f_out=results(:,1);
% p_out=results(:,2);
% q_out=results(:,3);

bgnd_floor=ones(length(f),1)*BN;
noiseline=bgnd_floor;

%----------------------------------------------------------------------
% Apply Lorentzian fit to determine resonance parameters
function [fit_results, back_noise]=find_resonances(f,p)

norm_factor=1/max(p);   % normalize to help calculation accuracy
p=p*norm_factor;

% Parameters:
% fc=Center frequency
% A=Peak power
% Q=Quality factor
% BN = Background noise (noise variance)

% Fitting Model: Lorentzian function
fdiff=3;
Qmin=1;
Qmax=20;
Amin=0;
Amax=2*max(p);
Qstart=5;
BNmin=0;
BNmax=max(p);
BNstart=0;

ainits=[];
ferr=0.5;
for i=1:length(f_res)
    iii=find((f>f_res(i)-ferr) & (f<f_res(i)+ferr));
    ainits=[ainits;mean(p(iii))];   % set init power value
end
% initparams=[f_res', ones(modes,1), repmat(Qstart,modes,1)];
initparams=[f_res', ainits, repmat(Qstart,modes,1)];
initparams=reshape(initparams,1,modes*3);
low_params=reshape([f_res-fdiff;repmat(Amin,1,modes);repmat(Qmin,1,modes)],1,modes*3);
up_params=reshape([f_res+fdiff;repmat(Amax,1,modes);repmat(Qmax,1,modes)],1,modes*3);

initparams=[initparams,BNstart];
low_params=[low_params,BNmin];
up_params=[up_params,BNmax];

% 'TolFun',1e-8,'TolX',1e-8,
% 'MaxFunEvals',1000,'MaxIter',1000,
opts = fitoptions('Method','Nonlinear',...%'Robust','Bisquare',...
'lower',low_params,'upper',up_params,'startpoint',initparams);

modelstring='';
for ii=1:modes
% modelstring=strcat(modelstring,sprintf('(A%d/(1+4*(x-fc%d)^2/B%d^2))',ii,
% ii,ii));
modelstring=...
    strcat(modelstring,sprintf('(A%d/(1+4*Q%d^2*(x/fc%d-1)^2))',ii,ii,ii));

if ii<modes; modelstring=strcat(modelstring,'+'); end;
end
% Append background noise level
modelstring=[modelstring,'+BN'];

coeffs=cell(1,modes*3+1);
for j=1:modes
coeffs{(j-1)*3+1}=sprintf('fc%d',j);
coeffs{(j-1)*3+2}=sprintf('A%d',j);
coeffs{(j-1)*3+3}=sprintf('Q%d',j);
end
coeffs{modes*3+1}='BN';

fitmodel=fittype(modelstring,...
	'coeff',coeffs,...
	'options',opts);

[fresult,gof,out]=fit(f,p,fitmodel);
%--------------------------------------------------------------------------

fitline=fresult(f)/norm_factor;
fit_results=coeffvalues(fresult);
back_noise=fit_results(end)/norm_factor;
fit_results = reshape(fit_results(1:3*modes), [3,modes])';
% corect absolute power values
fit_results(:,2)=fit_results(:,2)/norm_factor;

end

end


