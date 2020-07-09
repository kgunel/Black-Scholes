function fcost = Cost(xval,tval,p)
% This function computes values of the Cost Function defined as
% E(p)=(1/2)*Sum_x*Sum_t (eij)^2
% eij=U_t+(1/2)*x^2*A^2*U_xx+B*x*U_x-C*U,
% U(x,t) is the trial function defined as
% U(x,t)=(t/Tf)*P(x)+x*(Tf-t)*N(x,t;p)
global A B C E
Nx = length(xval);
Nt = length(tval);
Tf=tval(Nt);
xA=xval(1);
fE=max(xval-E,0);
fdPx=dP_x(xval);
fNet=Net(xval,tval,p);
fdNet_t=dNet_t(xval,tval,p);
fdNet_x=dNet_x(xval,tval,p);
fd2Net_xx=d2Net_xx(xval,tval,p);
fcost =0;
for j=1:Nt
    tj=tval(j);
    for i=1:Nx
        xi=xval(i);
        fdtrial_t=fE(i)/Tf-(xi-xA)*fNet(i,j)+(xi-xA)*(Tf-tj)*fdNet_t(i,j);
        ftrial=tj*fE(i)/Tf+(xi-xA)*(Tf-tj)*fNet(i,j);
        fdtrial_x=tj*fdPx(i)/Tf+(Tf-tj)*fNet(i,j)+(xi-xA)*(Tf-tj)*fdNet_x(i,j);
        fd2trial_xx=2*(Tf-tj)*fdNet_x(i,j)+(xi-xA)*(Tf-tj)*fd2Net_xx(i,j);
        fcost =  fcost+0.5*(fdtrial_t+0.5*xi^2*A^2*fd2trial_xx+B*xi*fdtrial_x-C*ftrial)^2;
    end
end


