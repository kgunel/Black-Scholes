% This program solves the Black-Scholes problem defined as
% U_t+(1/2)*x^2*A^2*U_xx+B*x*U_x-C*U=0, 0<x<1, 0<t<Tf,
% U(0,t)=0, left boundary condition
% U(x,Tf)=P(x),final time condition, P(x)=max{x-E,0},
% by using Artificial Neural Networks (ANN) and
% by Particle Swarm Optimization (PSO)
clc;
clear;
close;
global A B C E
%% Problem definition
CostFunction =@(x,t,p)  Cost(x,t,p);            % Cost function
A=0.1;B=0.12;C=B;                               % Coefficients of the equation
xA=70; xB=130;                                  % Left and right boundary points
Tf=1;                                           % Final time point
E=100;                                          % Strike price
N=5;                                            % Number of neurons 
%Nt=21;Nx=21;                                   % Number of nodes
%xval=linspace(xA,xB,Nx);                       % Spatial vector x
%tval=linspace(0,Tf,Nt);                        % Time vector t
%Uexc = exact_u(xval,tval);                     % Exact solution

%% Parameters of Pso
MaxIt = 1000;                                    % Maximum number of iterations
nPop = 100;                                     % Population size ( swarm size )
ww=1;                                           % Inertia coefficient
wdamp=0.99;                                     % Damping parameter of intertia coefficient
c1 = 1.5;                                       % Personal weight
c2 = 1.5;                                       % Social weight
nVar = 4*N;                                     % Number of unknown variables in ANN-algorithm
VarSize=[1 nVar];                               % matrix size of unknown variables
VarMin = -1;                                    % Lower bound of unknown variables
VarMax = 1;                                     % Upper bound of unknown variables
maxTrials = 25;                                 % Maximum number of trials

%% Initialization
empty_particle.position = [];
empty_particle.velocity = [];
empty_particle.cost = [];
empty_particle.best.position = [];
empty_particle.best.cost = [];

empty_sol.N = 5;
empty_sol.BestCosts = [];
empty_sol.Cost = [];
empty_sol.p = [];
empty_sol.Uapp = [];
empty_sol.errorL1 = [];
empty_sol.errorL2 = [];
empty_sol.RelErrL2 = [];

for Nx = [11,21]
    for Nt = [11,21]
        fprintf('N_x = %2d \t N_t = %2d\n',Nx,Nt);
        
        xval=linspace(xA,xB,Nx);                        % Spatial vector x
        tval=linspace(0,Tf,Nt);                         % Time vector t
        Uexc = exact_u(xval,tval);                      % Exact solution
        
        [mx,nx] = size(xval) ;
        [mt,nt] = size(tval) ;
        P = 0.7 ;
        xTrain = xval(1:round(P*nx))' ; 
        xTest = xval(round(P*nx)+1:end)';
        tTrain = tval(1:round(P*nt))' ; 
        tTest = tval(round(P*nt)+1:end)';
        
       % xTrain=Training_xval(1:nx-1).'; % Note transposes
       % tTrain=Training_tval(1:nt-1).'; % Note transposes
        
       [uexc,selectedInput,desiredOutput]=exact_u_v2(xTrain,tTrain);
        
       % XTest=Testing(:,1:n-1).';
       % YTest=Testing(:,n).';
        
       c = cvpartition(numel(desiredOutput),'Holdout', 0.25);
       hiddenLayerSize = optimizableVariable('hiddenLayerSize',[3,50], 'Type', 'integer');
       minfn = @(z)wrapFitNet(selectedInput,desiredOutput, 'CVPartition', c, ...
             'hiddenLayerSize',z.hiddenLayerSize);

         resultBayesOpt = bayesopt(minfn,hiddenLayerSize,'IsObjectiveDeterministic',false,...
             'MaxObjectiveEvaluations',100,...
             'AcquisitionFunctionName','probability-of-improvement',...
             'PlotFcn',[]);
         %'expected-improvement-plus');
         
         %[~,ind] = min([results.EstimatedObjectiveMinimumTrace]);
         N=resultBayesOpt.XAtMinEstimatedObjective.hiddenLayerSize;   % Number of neurons
         nVar = 4*N;                                     % Number of unknown variables in ANN-algorithm
         VarSize=[1 nVar];                               % matrix size of unknown variables

        
        sol = repmat(empty_sol, maxTrials, 1);
        for trialId = 1:maxTrials
            
            %fprintf('\n\n-----------------------------------------------------\n');
            fprintf('\t\t Trial Id : %d\n', trialId);
            T=cputime;
            ww=1;
            % Generate initial population
            particle = repmat(empty_particle, nPop, 1);
            
            % Initialize global best
            GlobalBest.cost = inf;
            
            for i=1:nPop
                % Generate random solution
                particle(i).position = unifrnd(VarMin, VarMax, VarSize);
                
                % Initialize velocity
                particle(i).velocity = zeros(VarSize);
                
                % Evaluation
                particle(i).cost = CostFunction (xval,tval,particle(i).position) ;
                
                % Update the personel best
                particle(i).best.position = particle(i).position ;
                particle(i).best.cost = particle(i).cost ;
                
                % Update the global best
                if particle(i).best.cost < GlobalBest.cost
                    GlobalBest = particle(i).best;
                end
            end
            
            % Array to hold the best cost value on each iteration
            BestCosts = zeros(MaxIt,1);
            
            %% Main loop of Pso
            for it=1:MaxIt
                
                for i=1:nPop
                    particle(i).velocity =  ww*particle(i).velocity...
                        +c1*rand(VarSize).*(particle(i).best.position-particle(i).position)...
                        +c2*rand(VarSize).*(GlobalBest.position-particle(i).position);
                    
                    particle(i).position = particle(i).position +  particle(i).velocity;
                    
                    particle(i).cost = CostFunction ( xval,tval,particle(i).position) ;
                    
                    if particle(i).cost < particle(i).best.cost
                        
                        particle(i).best.position = particle(i).position;
                        particle(i).best.cost = particle(i).cost;
                        
                        % Update the global best
                        if particle(i).best.cost < GlobalBest.cost
                            GlobalBest = particle(i).best;
                        end
                    end
                end
                
                % Store best cost value
                BestCosts(it) = GlobalBest.cost;
                
                % Display iteration information
                %fprintf('\t\tIteration %4d : Best Cost = %4.3e\n', it, BestCosts(it) );
                
                % Damping intertia coefficient
                ww = ww*wdamp;
            end
            
            
            %% Construct approximate solution
            p_opt = GlobalBest.position;
            Uapp = trial_u(xval,tval,p_opt);
            
            errorL1=(sum(sum(abs(Uexc-Uapp))))/(Nx*Nt);
            errorL2=norm(Uexc-Uapp,'fro')/(sqrt(Nx*Nt));
            RelErrL2=errorL2/norm(Uexc,'fro')/(sqrt(Nx*Nt));
            
            sol(trialId).N = N;
            sol(trialId).BestCosts = BestCosts;
            sol(trialId).Cost = GlobalBest.cost;
            sol(trialId).p = p_opt;
            sol(trialId).Uapp = Uapp;
            sol(trialId).errorL1 = errorL1;
            sol(trialId).errorL2 = errorL2;
            sol(trialId).RelErrL2 = RelErrL2;
            sol(trialId).elapsedTime = cputime - T;
            
        end
        Costs = [sol(1:maxTrials).Cost];
        [~,minId] = min(Costs);
        
        BestCosts = sol(minId).BestCosts ;
        p_opt = sol(minId).p;
        Uapp =  sol(minId).Uapp;
        errorL1 = sol(minId).errorL1;
        errorL2 = sol(minId).errorL2;
        RelErrL2 = sol(minId).RelErrL2;
        %errorL3 = sol(minId).errorL3;
        
        %% Results
        fileName = ['Results\PSO_' num2str(Nx) '_'  num2str(Nt) '.txt'];
        fid = fopen(fileName,'w+');
        
        minErrorL1 = min([sol(1:maxTrials).errorL1]);
        worstErrorL1 = max([sol(1:maxTrials).errorL1]);
        meanErrorL1 = mean([sol(1:maxTrials).errorL1]);
        stdErrorL1 = std([sol(1:maxTrials).errorL1]);
        
        minErrorL2 = min([sol(1:maxTrials).errorL2]);
        worstErrorL2 = max([sol(1:maxTrials).errorL2]);
        meanErrorL2 = mean([sol(1:maxTrials).errorL2]);
        stdErrorL2 = std([sol(1:maxTrials).errorL2]);
        
        minRelErrL2 = min([sol(1:maxTrials).RelErrL2]);
        worstRelErrL2 = max([sol(1:maxTrials).RelErrL2]);
        meanRelErrL2 = mean([sol(1:maxTrials).RelErrL2]);
        stdRelErrL2 = std([sol(1:maxTrials).RelErrL2]);
        
        meanTime = mean([sol(1:maxTrials).elapsedTime]);
        stdTime = std([sol(1:maxTrials).elapsedTime]);
        
        fprintf('\n\n');
        pm = 177;               % ASCII code for plus minus symbol
        fprintf('Results for %d neurons in the hidden layer\n',N);
        disp('******************** Errors ********************')
        fprintf('Min of AbsErrorL1 = %0.3e\n', minErrorL1);
        fprintf('Worst of AbsErrorL1 = %0.3e\n', worstErrorL1);
        fprintf('Mean of AbsErrorL1 = %0.3e %c %0.3e\n\n', meanErrorL1, pm, stdErrorL1);
        
        fprintf('Min of AbsErrorL2 = %0.3e\n', minErrorL2);
        fprintf('Worst of AbsErrorL2 = %0.3e\n', worstErrorL2);
        fprintf('Mean of AbsErrorL2 = %0.3e %c %0.3e\n\n', meanErrorL2, pm, stdErrorL2);
        
        fprintf('Min of RelErrorInf = %0.3e\n', minRelErrL2);
        fprintf('Worst of RelErrorInf = %0.3e\n', worstRelErrL2);
        fprintf('Mean of RelErrorInf = %0.3e %c %0.3e\n\n', meanRelErrL2, pm, stdRelErrL2);
        
        fprintf('Mean of Elapsed Time = %0.3e %c %0.3e seconds\n\n', meanTime, pm, stdTime);
        
        
        fprintf(fid,'Results for %d neurons in the hidden layer\n',N);
        fprintf(fid,'Min of AbsErrorL1 = %0.3e\n', minErrorL1);
        fprintf(fid,'Worst of AbsErrorL1 = %0.3e\n', worstErrorL1);
        fprintf(fid,'Mean of AbsErrorL1 = %0.3e %c %0.3e\n\n', meanErrorL1, pm, stdErrorL1);
        
        fprintf(fid,'Min of AbsErrorL2 = %0.3e\n', minErrorL2);
        fprintf(fid,'Worst of AbsErrorL2 = %0.3e\n', worstErrorL2);
        fprintf(fid,'Mean of AbsErrorL2 = %0.3e %c %0.3e\n\n', meanErrorL2, pm, stdErrorL2);
        
        fprintf(fid,'Min of RelErrorInf = %0.3e\n', minRelErrL2);
        fprintf(fid,'Worst of RelErrorInf = %0.3e\n', worstRelErrL2);
        fprintf(fid,'Mean of RelErrorInf = %0.3e %c %0.3e\n\n', meanRelErrL2, pm, stdRelErrL2);
        
        fprintf(fid,'Mean of Elapsed Time = %0.3e %c %0.3e seconds\n\n', meanTime, pm, stdTime);
        
        fclose(fid);
        
        results.ErrorL1.min = minErrorL1;
        results.ErrorL1.worst = worstErrorL1;
        results.ErrorL1.mean = meanErrorL1;
        results.ErrorL1.std = stdErrorL1;
        
        results.ErrorL2.min = minErrorL2;
        results.ErrorL2.worst = worstErrorL2;
        results.ErrorL2.mean = meanErrorL2;
        results.ErrorL2.std = stdErrorL2;
        
        results.RelErrorL2.min = minRelErrL2;
        results.RelErrorL2.worst = worstRelErrL2;
        results.RelErrorL2.mean = meanRelErrL2;
        results.RelErrorL2.std = stdRelErrL2;
        
%         %% Plots
%         fig1 = figure;
%         % axis([xA xB 0 Tf 0 max(max(Uexc))])
%         subplot(1,2,1);mesh(xval,tval,Uexc);
%         xlabel('$x$','Interpreter','latex')
%         ylabel('$t$','Interpreter','latex')
%         zlabel('Exact Solution','Interpreter','latex')
%         % axis([xA xB 0 Tf 0 max(max(Uapp))])
%         subplot(1,2,2);mesh(xval,tval,Uapp)
%         xlabel('$x$','Interpreter','latex')
%         ylabel('$t$','Interpreter','latex')
%         zlabel('Numerical Solution','Interpreter','latex')
%         print(fig1,'Fig1.eps','-depsc','-r300');
%         print(fig1,'Fig1.jpg','-djpeg','-r300');
%         
%         fig2=figure;
%         subplot(1,2,1);plot(xval,Uexc(1,:),'k-',xval,Uapp(1,:),'r-.','LineWidth',1.5)
%         xlabel('$x$','FontSize',12,'Interpreter','latex')
%         ylabel('Numerical solution at $t=0$','FontSize',12,'Interpreter','latex')
%         legend('Exact Solution','Numerical solution','Location','northwest')
%         subplot(1,2,2);semilogy(BestCosts(1:200), 'linewidth',2)
%         xlabel('Iterations','FontSize',12,'Interpreter','latex')
%         ylabel('Best Cost','FontSize',12,'Interpreter','latex')
%         grid on
%         print(fig2,'Fig2.eps','-depsc','-r300');
%         print(fig2,'Fig2.jpg','-djpeg','-r300');
        
        save(['Results\PSO_' num2str(Nx) '_'  num2str(Nt) '.mat'],'sol','BestCosts','p_opt','Uexc','Uapp','results');
    end
end
