M = 0.536; %0.536   190g for linear bearing*2 + 11.4g*4 bearings + 60g plastic + 50g metal rod
m = 0.8; %0.8  Approximately 800grams
b_c = 3; %0.1
b_m = 0.1;
l = 0.3; %0.3
J_p = 0.0006; %0.006
g = -9.81;
d_p = 0.0127; %12.7 mm
J_m = 0.0006;
K_V = 17.65; %Line relating voltage and speed; vel = 17.65*(v_in) - 29.1 
K_T = 0.118; %0.1181 Nm/A %0.3
R_a = 2.24; %12.3 %1.5
K_M = K_T/sqrt(R_a);

h = 0.002; %Sampling period
Ts = 0.002;

global h_1p;
global h_1c;
global r_p;
global r_c;
global c_p;
global c_c;
global beta_01p;
global beta_02p;
global beta_03p;
global beta_01c;
global beta_02c;
global beta_03c;
global b_0;

h_1p = 0.002; %0.015
h_1c = 0.002;
r_p = 10; %0.15
r_c = 10;
c_p = 1; %100
c_c = 1;
b_0 = 0.1; %0.1

beta_01p = 1;
beta_02p = 1/3*h;
beta_03p = 2/64*h^2;

beta_01c = 1;
beta_02c = 1/3*h;
beta_03c = 2/64*h^2;

global solution;
solution = zeros(100,3);



%%%%%%%%%% Optimizer Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pop_size = 100; %Population Size
num_obj = 6; %Number of objectives
num_div = 10; %Number of divisions
dim = 13; %Dimension of Problem
global zeta_c; %Crossover Rate
global zeta_m; %Mutation Rate
zeta_c = 20; % Recommended by "Analyzing Mutation Schemes for Real-Parameter Genetic Algorithms" Deb and Deb
zeta_m = 20; % '20'
generation = 1; %Number of generations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




q = ((J_p*(M+m))+(M*m*l^2));
au_22 = -((J_p+m*l^2)*b_c)/q;
au_23 = (2*(m*l)^2*g)/(d_p*q);
au_42 = -(m*l*d_p*b_c)/(2*q);
au_43 = (m*g*l*(M+m))/q;
bu_2 = (2*(J_p+m*l^2)*K_M)/(d_p*R_a*q); %Changed KT to K_m
bu_4 = (m*l*K_M)/(R_a*q); %Changed K_T to K_M

A = [0 1 0 0; 0 au_22 au_23 0; 0 0 0 1; 0 au_42 au_43 0];
B = [0; bu_2; 0; bu_4];
C = [1 0 0 0; 0 0 1 0];


sys = ss(A,B,C,0);

sysd = c2d(sys,h);

global Ad;
global Bd;
global Cd;

Ad = sysd.A;
Bd = sysd.B;
Cd = sysd.C;

sim('Inv_Pend_Sim_Compressed', 'CaptureErrors', 'on');


range = ones(dim, 2); %Create range of variables for evaluation in
%the form of [max min; max min; ...]


range(1:2,1) = 0.01; %h_1p,c max
range(1:2,2) = 0.00001; %h_1p,c min
range(3:4,1) = 100; %r_p,c max
range(3:4,2) = 0.01; %r_p,c min
range(5:6,1) = 1.5; %c_p,c max
range(5:6,2) = 0.5; %c_p,c min
range(7,1) = 1; %b_0 max
range(7,2) = 0.0001; %b_0 min
range(8:9,1) = 2; %beta_01p,c max
range(8:9,2) = 0.5; %beta_01p,c min
range(10:11,1) = 0.001; %beta_02p,c max
range(10:11,2) = 0.000000001; %beta_02p,c min
range(12:13,1) = 0.001; %beta_03p,c max
range(12:13,2) = 0.000000001; %beta_03p,c min

%Initialize population
parent_pop = zeros(pop_size,dim); %Parent Population
variation_pop = zeros(pop_size,dim); %Variant Population
fitness_extremes = zeros(num_obj, 2); %Formatted as [max, min]
landmarks = zeros(pop_size,dim,30);
stored_fitness = ones(20,num_obj, 501);

%\\\\\\\   Initialize Parent Population 
parent_pop = rand(pop_size,dim);
for i = 1:dim
    %Map random numbers to variable range = rand*(max - min) + min
    parent_pop(:,i) =  parent_pop(:,i).*(range(i,1) - range(i,2)) +  range(i,2);
end 
%///////////////////////////////////////


%\\\\\\\\\\\\\\\   Reference Point Set
% Based on normalized hyperplane
num_combs = nchoosek((num_obj + num_div - 1), num_div); %Defines number of possible combinations
normalized_vec = linspace(0,1, num_div + 1);  
A_ref = permn(normalized_vec, num_obj); %Get All Permutations
C_ref = find(sum(A_ref') == 1); %Find Permutations with sum = 1
% From "Normal-Boundary Intersection: A New Method for Generating the
% Pareto Surface in Nonlinear Multicriteria Optimization Problems"

%RPset = zeros(num_combs, num_obj); %Referece Point Set
RPdensity = zeros(num_combs,1); %Reference Point Density
%Generate Reference Points
for i = 1:sum(sum(A_ref') == 1)
    RPset(i,:) = A_ref(C_ref(i),:);
end
%//////////////////////////////////////


% Run the optimizer RPD-NSGA-2
while generation < 500 %Run for 1000 Generations
   next_gen_pop = zeros(pop_size,dim); %Zero Child Population
   variation_pop = variation(parent_pop, range); %Vary Parent Population
   merged_pop = union(parent_pop, variation_pop, 'rows'); %Merge Parent and Variant Populations
   fitness = zeros(size(merged_pop,1), num_obj); %Zero Fitness Values
   %Find fitness for each objective
   for i = 1:num_obj
       fitness(:,i) = objective(merged_pop, i);
  
       %Find extreme fitness values for each objective for normalization
       fitness_extremes(i, 2) = min(fitness(:,i));
       fitness_extremes(i, 1) = max(fitness(:,i));
   end
   
   stored_fitness(1:size(fitness,1),:,generation) = fitness; 
   %Normalize the fitness to [0,1]
   fitness = normalize(fitness, fitness_extremes);
   
   %Bias Most Important Fitness Values
   for i = 1:num_obj
       if i > 2
            fitness(:,i) = fitness(:,i) + i*10;
       end
   end
   
   %d_2 is the euclidean distance to the nearest reference point 
   d_2 = zeros(size(fitness,1), 3); % [value, index, RPdensity]
   d_2(:,1:2) = calc_d2(fitness, RPset);
   
   %d_1 is the distance from the origin to the foot of the normal
   d_1 = zeros(size(fitness,1), 1); % [value]
   d_1 = calc_d1(fitness, RPset, d_2);
   
   %Calculate density with respect to each reference point
   for i = 1:num_combs
      RPdensity(i) = sum(d_2(:,2) == i); 
   end
   
   %Associate Density to each individual
   d_2(:,3) = RPdensity(d_2(:,2));
   
   %favour extremes by setting density to zero

   for i = 1:num_obj
      [o idx] = min(fitness(:,i));
      d_2(idx, 3) = 0.001; %Set RPdensity of extremes to 0
   end

   %Calculate non-RPD domination ranks
   fronts = non_RPD_dominated_sorting(merged_pop, fitness, d_1, d_2);
   
   %crowd_d = Crowd(fitness);
   %crowd_d = prod(crowd_d')';
   %sol_location = find(crowd_d == max(crowd_d));
   %Create Child Population with members of the best performing fronts
   for i = 1:pop_size
       [val idx] = min(fronts);
       num_sol_in_front = size(find(fronts == val),1);
       if num_sol_in_front <= (pop_size + 1 - i)
            next_gen_pop(i,:) = merged_pop(idx,:);
            fronts(idx) = pop_size+1; %Makes sure solution not chosen again
       elseif num_sol_in_front > (pop_size - i)%Truncation of last front
            
            sol_location = find(fronts == val);
            
            last_f_best = d_2(sol_location(1),1);
            %last_f_best = crowd_d(sol_location(1),1);
            best_loc = sol_location(1);
            for j = 1:(size(find(fronts == val),1)-1)
            %for j = 1:(size(find(crowd_d == max(crowd_d)),1)-1)
                if d_2(sol_location(j+1),1) < best_loc
                %if crowd_d(sol_location(j+1),1) > best_loc
                    last_f_best = d_2(sol_location(j+1),1);
                    %last_f_best = crowd_d(sol_location(j+1),1);
                    best_loc = sol_location(j+1);
                end
            end
            next_gen_pop(i,:) = merged_pop(best_loc,:);
       end
   end
  
   parent_pop = next_gen_pop;
   
   generation = generation + 1
   %Save landmark populations for improvement comparison
   if rem(generation, 25) == 0
       gen_index = generation/25;
       landmarks(:,:,gen_index) = parent_pop;
   end
   
   
   
end






function rp_fronts = non_RPD_dominated_sorting(merged_pop, fitness, d_1, d_2)
    size_pop = size(merged_pop,1);
    num_obj = size(fitness,2);
    %p_dom = true(size_pop, size_pop);
    p_dom = false(size_pop, size_pop);
    total_p_dom = true(size_pop, size_pop);
    rp_dom = true(size_pop, size_pop);
    p_fronts = zeros(size_pop,1);
    rp_fronts = zeros(size_pop,1);
    %fronts = zeros(size_pop,2);
    
    for i = 1:size_pop
        for j = 1:num_obj
            p_dom(i,:) = p_dom(i,:) | logical(fitness(i,j) < fitness(:,j))';
        end
        %rp_dom(i,:) = rp_dom(i,:) & p_dom' ; %Pareto Dom. Condition 1.)
        rp_a = logical(d_2(i,2) == d_2(:,2)) & logical(d_1(i) < d_1(:));
        rp_b = logical(d_2(i,2) ~= d_2(:,2)) & logical(d_1(i) < d_1(:)) & logical(d_2(i,3) < d_2(:,3));
        rp_dom(i,:) = rp_a' | rp_b';
        
    end
    i = 1;
    count = 1;
    selected_sol = zeros(1,size_pop);
    p_dom_check = p_dom;
    %Find Pareto Fronts usind non-pareto criteria
    while(min(sum(p_dom_check')) ~= size_pop)
        sum_p_dom = sum(p_dom_check');
        %p_dom_unique = unique(sum_p_dom);
        %num_fronts = size(p_dom_unique,2); 
        %C = find(sum_p_dom == p_dom_unique(i));
        C = find(sum_p_dom == min(sum_p_dom));
        for j = 1:size(C,2)
            p_fronts(C(j)) = i;
            p_dom_check(:, C(j)) = false(size_pop,1); %Reset Dominance for last front
            selected_sol(count) = C(j);
            count = count + 1;
        end
        for j = 1:count-1
            p_dom_check(selected_sol(j),:) = true(1,size_pop); %Exclude from future pf calc
        end
        i = i + 1;
    end
    p_fronts = (i) - p_fronts;
    %Find RP-dominated Pareto Front
   
    for i = 1:size_pop
        rp_dom(i,:) = (logical(p_fronts(i) < p_fronts(:)))' | (logical(p_fronts(i) == p_fronts(:))' & rp_dom(i,:)); 
    end
    
    i = 1;
    count = 1;
    selected_sol = zeros(1,size_pop);
    rp_dom_check = rp_dom;
    %Find Pareto Fronts usind non-pareto criteria
    while(min(sum(rp_dom_check')) ~= size_pop)
        sum_rp_dom = sum(rp_dom_check');
        %p_dom_unique = unique(sum_p_dom);
        %num_fronts = size(p_dom_unique,2); 
        %C = find(sum_p_dom == p_dom_unique(i));
        C = find(sum_rp_dom == min(sum_rp_dom));
        for j = 1:size(C,2)
            rp_fronts(C(j)) = i;
            rp_dom_check(:, C(j)) = false(size_pop,1); %Reset Dominance for last front
            selected_sol(count) = C(j);
            count = count + 1;
        end
        for j = 1:count-1
            rp_dom_check(selected_sol(j),:) = true(1,size_pop); %Exclude from future pf calc
        end
        i = i + 1;
    end
    rp_fronts = (i) - rp_fronts;
    %fronts = [p_fronts rp_fronts];
    
end


function sol = objective(m_pop, idx)
    pop_size = size(m_pop,1);
    global h_1p;
    global h_1c;
    global r_p;
    global r_c;
    global c_p;
    global c_c;
    global b_0;
    global beta_01p;
    global beta_01c;
    global beta_02p;
    global beta_02c;
    global beta_03p;
    global beta_03c;
    global solution;
  
    global Ad;
    global Bd;
    global Cd;
    
    %sol = zeros(size(m_pop,1),1);
        switch idx
            case 1
               %sim_results = ones(5001,6)*30;
               for i = 1:pop_size
    
                    
                    h_1p = m_pop(i,1);
                    h_1c = m_pop(i,2);
                    
                    r_p = m_pop(i,3);
                    r_c = m_pop(i,4);
                    
                    c_p = m_pop(i,5);
                    c_c = m_pop(i,6);
                    
                    b_0 = m_pop(i,7);
                    
                    beta_01p = m_pop(i,8);
                    beta_01c = m_pop(i,9);
                    
                    beta_02p = m_pop(i,10);
                    beta_02c = m_pop(i,11);
                    
                    beta_03p = m_pop(i,12);
                    beta_03c = m_pop(i,13);
          
                    %Run Simulation
                    res = sim('Inv_Pend_Sim_Compressed', 'CaptureErrors', 'on');
                    %This is the input signal to be minimized
                    
                    
                    if(size(res.ErrorMessage,1) == 0) %If no error
                        
                        %Pendulum Tracking Error
                        sol(i,1) = sum(abs(res.ADRC_Simulation.Data(:,1))); 
                        solution(i,1) = sol(i,1); 
                        
                        %Cart Tracking Error (Ref - Actual)
                        solution(i,2) = sum(abs(res.ADRC_Simulation.Data(:,4)-res.ADRC_Simulation.Data(:,2))); 
                        
                        %Control Effort
                        %solution(i,3) = sum(abs(res.ADRC_Simulation.Data(:,3))); 
                        
                        %Rise Time
                        rise_time = find(res.ADRC_Simulation.Data(126:4000,2) > 0.1,1,'first');
                        if rise_time >= 1
                            solution(i,3) = res.ADRC_Simulation.Time(126 + rise_time);
                        else 
                            solution(i,3) = 15 - 0.01*i; %Max Time
                        end
                        
                        %Steady State Error (last 0.5s before disturbance)
                        solution(i,4) = sum(abs(res.ADRC_Simulation.Data(2500:2749,4)-res.ADRC_Simulation.Data(2500:2749,2)));
                        
                        %Percent Overshoot
                        if (max(res.ADRC_Simulation.Data(126:2749,2)) > 0.1) %If an overshoot exists
                            solution(i,5) = abs(max(res.ADRC_Simulation.Data(126:2749,2)) - 0.1); %Cart Overshoot
                        elseif (solution(i,4) < 1) %If the solution has little steady state error
                            solution(i,5) = 0; %Set overshoot to zero
                        else  %Set if no overshoot and poor steady state response
                            solution(i,5) = 5 - 0.01*i; %Penalize weak controller
                        end
                        
                        %Settling Time
                        %find last time where abs(ref - cart_pos) < 5% ref 
                        settling_time = find(abs(res.ADRC_Simulation.Data(126:2749,4) - res.ADRC_Simulation.Data(126:2749,2)) > 0.005,1,'last');
                        if settling_time >= 2500 %If not settled in 5.25 seconds
                            solution(i,6) = 20 - 0.01*i; %Penalize for not settling
                        elseif settling_time >= 1
                            solution(i,6) = res.ADRC_Simulation.Time(126 + settling_time);
                        else
                            solution(i,6) = 15 - 0.01*i;
                        end
                            
                            
                    else
                        %If the simulation returns an error: set fitness
                        %something large
                        sol(i,1) = 9999 - i; 
                        solution(i,1) = 9999 - i; %Pendulum Tracking
                        solution(i,2) = 9999 - i; %Cart Tracking
                        solution(i,3) = 9999 - i; %Control Effort
                        solution(i,4) = 9999 - i; %Rise Time
                        solution(i,5) = 9999 - i; %Steady State Error
                        solution(i,6) = 9999 - i; %Percent Overshoot
                        solution(i,7) = 9999 - i; %Settling Time
                    end
                end
                
            case 2
               for i = 1:pop_size
                    sol(i,1) = solution(i,2);
                end
            case 3
               for i = 1:pop_size
                    sol(i,1) = solution(i,3);       
               end
            case 4
                for i = 1:pop_size
                    sol(i,1) = solution(i,4);       
                end
            case 5
                for i = 1:pop_size
                    sol(i,1) = solution(i,5);       
                end
            case 6
                for i = 1:pop_size
                    sol(i,1) = solution(i,6);       
                end
            case 7
                for i = 1:pop_size
                    sol(i,1) = solution(i,7);       
               end
     
                
         end
              
end

%Calculates RPDistance D1

function d_1 = calc_d1(fit, ref, d_2)
    [pop_size, num_f] = size(fit);
    d_1 = zeros(pop_size,1);
    for n = 1:pop_size 
        d_1(n) = dot(fit(n,:), ref(d_2(n,2),:));
    end
end

%Calculates RPDistance D2

function d_2 = calc_d2(fit, RPset)
    [pop_size, num_f] = size(fit);
    d_2 = zeros(pop_size,2);
    for n = 1:pop_size 
        d2_dist = sqrt(sum((fit(n,:)-RPset)'.^2)');
        [d_2(n,1), d_2(n,2)] = min(d2_dist);
    end
    
end

%Normalizes Fitness Values

function norm = normalize(pop, extremes)
    norm = zeros(size(pop,1), size(pop,2));
    for k = 1:size(pop,2)
       norm(:,k) = (pop(:,k) - extremes(k,2))./(extremes(k,1) - extremes(k,2));
    end
end

%Crossover and Mutation
function Q_t = variation(P_t, range)
    global zeta_c;
    global zeta_m;
    pop_size = size(P_t,1);
    dim = size(P_t,2);
        
    Q_t = zeros(pop_size, dim);
    p_1 = zeros(1,dim); %Parent 1
    p_2 = zeros(1,dim); %Parent 2
    c_1 = zeros(1,dim); %Child 1
    c_2 = zeros(1,dim); %Child 2
    for i = 1:2:pop_size
        %Randonmly select 2 parents
        p_1 = P_t(ceil(rand*pop_size), :);
        p_2 = P_t(ceil(rand*pop_size), :);
        % Gaussian Selection
        %rand1 = (1/sqrt(2*pi*0.01))*e^(-(rand)^2/(2*0.01));
        %rand2 = (1/sqrt(2*pi*0.01))*e^(-(rand)^2/(2*0.01));
        %rand1 = 1 - exp(-(rand)^2/(2*0.01));
        %rand2 = 1 - exp(-(rand)^2/(2*0.01));
        %p_1 = P_t(ceil(rand1*pop_size), :);
        %p_2 = P_t(ceil(rand2*pop_size), :);
        %"An improved adaptive NSGA-II with multi-population algorithm 
        %"Analyzing Mutation Schemes for Real-Parameter Genetic Algorithms"
        %"Simulated Binary Crossover for Continuous Search Space"
        %"Real-coded Genetic Algorithms with Simulated Binary Crossover: Studies on Multimodal and Multiobjective Problems
        for j = 1:dim
            u_c = rand;
            u_m = rand;
            if u_c <= 0.5
                beta = (2*u_c)^(1/(zeta_c + 1)); 
            else
                beta = (2*(1 - u_c))^(1/(zeta_c + 1)); 
            end
            %Crossover
            c_1(j) = 0.5*((1 - beta)*p_1(j) + (1 + beta)*p_2(j));
            c_2(j) = 0.5*((1 + beta)*p_1(j) + (1 - beta)*p_2(j));
            %Polynomial Mutation
            
            if u_m <= 0.5
                delta_l = (2*u_m)^(1/(zeta_m + 1)) - 1;
                c_1(j) = c_1(j) + delta_l*(c_1(j) - range(j,2));
                c_2(j) = c_2(j) + delta_l*(c_2(j) - range(j,2));
            else
                delta_r = 1 - (2*(1 - u_m))^(1/(zeta_m + 1)); 
                c_1(j) = c_1(j) + delta_r*(range(j,1) - c_1(j));
                c_2(j) = c_2(j) + delta_r*(range(j,1) - c_2(j));
            end
            
        end
        Q_t(i,:) = c_1;
        Q_t(i + 1,:) = c_2;
    end
end


function [M, I] = permn(V, N, K)

% tested in Matlab 2016a
% version 6.1 (may 2016)
% (c) Jos van der Geest
% Matlab File Exchange Author ID: 10584
% email: samelinoa@gmail.com

narginchk(2,3) ;
if fix(N) ~= N || N < 0 || numel(N) ~= 1 
    error('permn:negativeN','Second argument should be a positive integer') ;
end
nV = numel(V) ;
if nargin==2 % PERMN(V,N) - return all permutations
    
    if nV==0 || N == 0
        M = zeros(nV,N) ;
        I = zeros(nV,N) ;
        
    elseif N == 1
        % return column vectors
        M = V(:) ;
        I = (1:nV).' ;
    else
        % this is faster than the math trick used for the call with three
        % arguments.
        [Y{N:-1:1}] = ndgrid(1:nV) ;
        I = reshape(cat(N+1,Y{:}),[],N) ;
        M = V(I) ;
    end
else % PERMN(V,N,K) - return a subset of all permutations
    nK = numel(K) ;
    if nV == 0 || N == 0 || nK == 0
        M = zeros(numel(K), N) ;
        I = zeros(numel(K), N) ;
    elseif nK < 1 || any(K<1) || any(K ~= fix(K))
        error('permn:InvalidIndex','Third argument should contain positive integers.') ;
    else
        
        V = reshape(V,1,[]) ; % v1.1 make input a row vector
        nV = numel(V) ;
        Npos = nV^N ;
        if any(K > Npos)
            warning('permn:IndexOverflow', ...
                'Values of K exceeding the total number of combinations are saturated.')
            K = min(K, Npos) ;
        end
             
        % The engine is based on version 3.2 with the correction
        % suggested by Roger Stafford. This approach uses a single matrix
        % multiplication.
        B = nV.^(1-N:0) ;
        I = ((K(:)-.5) * B) ; % matrix multiplication
        I = rem(floor(I),nV) + 1 ;
        M = V(I) ;
    end
end
end

