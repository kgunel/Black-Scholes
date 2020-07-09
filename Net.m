function fNet = Net(xx,tt,p)
% This function computes values of the Net function N(x,t;p)
% fN is output of neural network corrsponding to the inputs x and t
Nx=length(xx);Nt=length(tt);
M = length(p)/4; %number of neurons
alpha = p(1:M);
omegaA = p(M+1:2*M);
omegaB = p(2*M+1:3*M);
bias = p(3*M+1:4*M);
fNet=zeros(Nx,Nt);
for i=1:Nx
    x=xx(i);
    for j=1:Nt
        t=tt(j);
        for k=1:M
            zk= omegaA(k)*t+omegaB(k)*x+bias(k);
            fNet(i,j)=fNet(i,j)+alpha(k)*fsigma(zk);
        end
    end
end
end