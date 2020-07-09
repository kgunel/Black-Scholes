function fdNet_x = dNet_x(xx,tt,p)
% This function computes values of the spatial derivative of the Net function N(x,t;p)
Nx=length(xx);Nt=length(tt);
M = length(p)/4; %number of neurons
alpha = p(1:M);
omegaA = p(M+1:2*M);
omegaB = p(2*M+1:3*M);
bias = p(3*M+1:4*M);
fdNet_x=zeros(Nx,Nt); %outputs of neural network
for i=1:Nx
    x=xx(i);
    for j=1:Nt
        t=tt(j);
        for k=1:M
            zk= omegaA(k)*t+omegaB(k)*x+bias(k);
            fz=fsigma(zk);
            fdNet_x(i,j) = fdNet_x(i,j)+alpha(k)*omegaB(k)*fz*(1-fz);
        end
    end
end
end