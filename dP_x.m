function fdpx = dP_x(xx)
% This function computes values of the derivative of the function P(x) 
% defined as P(x)=U(x,Tf), here, Tf is the final time point.
global E
Nx=length(xx);
fdpx=zeros(1,Nx);
for i=1:Nx
    x=xx(i);
if x < E
    fdpx(i)= 0;
else
    fdpx(i)=1;
end
end