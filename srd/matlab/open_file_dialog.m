function open_file_dialog

% start - end time (sec)
t1 = 0;
t2 = 600;

persistent lastPath;
global spectrum_range;

spectrum_range = [3, 48];

if isempty(lastPath) 
    lastPath = 0;
end
defname = '';

if lastPath==0
	[f, p] = uigetfile([defname,'*.srd']);
else
	[f, p] = uigetfile([lastPath,'*.srd']);
end
    lastPath=p;
if f==0; return; end;

fname=[p,f];

[dt,fs,x,y] = get_srd_data(fname);

t=((0:(length(x)-1)).')/fs;
ii=find(t>=t1 & t<=t2);
t=t(ii);
x=x(ii);
if ~isempty(y)
    y=y(ii);
end

[F, Pxx, Pyy, L1, L2, R1, R2] = srd_spec(dt, fs, x, y);


plot_file(dt, t, x, y, F, Pxx, Pyy, L1, L2);

end


