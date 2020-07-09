function ftrial = trial_u(xx,tt,p)
global E
m = length(xx);
n = length(tt);
ftrial = zeros(n,m);
xA=xx(1);Tf=tt(end);
fNet=Net(xx,tt,p);
fPx=max(xx-E,0);
for i=1:m
for j=1:n
    ftrial(j,i)=tt(j)*fPx(i)/Tf+(xx(i)-xA)*(Tf-tt(j))*fNet(i,j);
end
end
