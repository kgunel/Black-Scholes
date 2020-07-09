function [u,inputs,desiredOutputs]=exact_u_v2(xx,tt)
global A B  E
Nx=length(xx);Nt=length(tt);
Tf=tt(end);
u=zeros(Nt,Nx);
inputs=zeros(2,Nt*Nx);
desiredOutputs=zeros(1,Nt*Nx);
k=1;
for i=1:Nx
    x=xx(i);
    for j=1:Nt-1
        t=tt(j);
        p1=log(x/E)+(B+A^2/2)*(Tf-t);
        pp=A*sqrt(Tf-t);
        p2=log(x/E)+(B-A^2/2)*(Tf-t);
        z1=p1/pp;z2=p2/pp;
        u(j,i)=x*0.5*erfc(-z1/sqrt(2))-0.5*E*exp(-B*(Tf-t))*erfc(-z2/sqrt(2));
        inputs(1,k) = x;
        inputs(2,k) = t;
        desiredOutputs(k) = u(j,i);
        k=k+1;
    end
end
u(Nt,:)=max(xx-E,0);