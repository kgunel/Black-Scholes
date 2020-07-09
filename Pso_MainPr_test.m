% This program solves the Black-Scholes problem defined as
% U_t+(1/2)*x^2*A^2*U_xx+B*x*U_x-C*U=0, 0<x<1, 0<t<Tf,
% U(0,t)=0, left boundary condition
% U(x,Tf)=P(x),final time condition, P(x)=max{x-E,0},
% by using Artificial Neural Networks (ANN) and
% by Particle Swarm Optimization (PSO)
clc;
clear;
close;

load('Results\PSO_41_81.mat');
global A B C E
%% Problem definition
CostFunction =@(x,t,p)  Cost(x,t,p);            % Cost function
A=0.1;B=0.12;C=B;                               % Coefficients of the equation
xA=70; xB=130;                                  % Left and right boundary points
Tf=1;                                           % Final time point
E=100;                                          % Strike price
N=5;                                            % Number of neurons
Nt=81;Nx=41;                                    % Number of nodes
Delta_x = 0.5; Delta_t = 0.1;
xval = xA:Delta_x:xB;                           % Spatial vector x
tval = 0: Delta_t:Tf;                           % Time vector t

Uexc = exact_u(xval,tval);                      % Exact solution
Uapp = trial_u(xval,tval,p_opt);                % Construct approximate solution

% Results
errorL1=(sum(sum(abs(Uexc-Uapp))))/(Nx*Nt);
errorL2=norm(Uexc-Uapp,'fro')/(sqrt(Nx*Nt));
RelErrL2=errorL2/norm(Uexc,'fro')/(sqrt(Nx*Nt));

fprintf('\n\n');
disp('******************** Errors ********************')
disp('  i    j   S_i   t_j      Err ');
disp('------------------------------------------------')
% Table
m = length(xval); n = length(tval);
for i=1:m
    if mod(i,20)==1
        for j=1:n
            if mod(j,5)==1
                Err = abs(Uexc(j,i) - Uapp(j,i));
                
                fprintf('%3d %4d %5d %5.1f %12.3e \n',i,j,xval(i),tval(j), Err);
            end
        end
    end
end

disp('**********************************************')
fprintf('AbsErrorL1 = %0.3e\n', errorL1);
fprintf('AbsErrorL2 = %0.3e\n', errorL2);
fprintf('RelErrorInf = %0.3e\n', RelErrL2);