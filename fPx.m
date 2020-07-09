function fpx = fPx(x)
% This function computes values of the function P(x) defined as P(x)=U(x,Tf);
% Here, Tf is the final time point.
EE=100;%sqrt(1/19);
fpx= max(x-EE,0);
end