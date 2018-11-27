% RPD-NSGA-II
% Brayden DeBoon, November 2018
% brayden.deboon@uoit.net

pop_size = 100; %Population Size
num_obj = 2; %Number of objectives
num_div = 20; %Number of divisions
dim = 30; %Dimension of Problem
global zeta_c; %Crossover Rate
global zeta_m; %Mutation Rate
zeta_c = 20; % Recommended by "Analyzing Mutation Schemes for Real-Parameter Genetic Algorithms" Deb and Deb
zeta_m = 20; % ''
generation = 0; %Number of generations

range = ones(dim, 2); %Create range of variables for evaluation in
%the form of [max min; max min; ...]

range(:,1) = range(:,1)*1; %Max is 1000
range(:,2) = range(:,2)*0; %Min is -1000

%Initialize population
parent_pop = zeros(pop_size,dim); %Parent Population
variation_pop = zeros(pop_size,dim); %Variant Population
fitness_extremes = zeros(num_obj, 2); %Formatted as [max, min]

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
A = permn(normalized_vec, num_obj); %Get All Permutations
C = find(sum(A') == 1); %Find Permutations with sum = 1
% From "Normal-Boundary Intersection: A New Method for Generating the
% Pareto Surface in Nonlinear Multicriteria Optimization Problems"

RPset = zeros(num_combs, num_obj); %Referece Point Set
RPdensity = zeros(num_combs,1); %Reference Point Density
%Generate Reference Points
for i = 1:sum(B)
    RPset(i,:) = A(C(i),:);
end
%//////////////////////////////////////


% Run the optimizer RPD-NSGA-2
while generation < 1000 %Run for 1000 Generations
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
   %Normalize the fitness to [0,1]
   fitness = normalize(fitness, fitness_extremes);
   
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
      d_2(idx, 3) = 0; %Set RPdensity of extremes to 0
   end
   
   %Calculate non-RPD domination ranks
   fronts = non_RPD_dominated_sorting(merged_pop, fitness, d_1, d_2, fitness_extremes);
   
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
            best_loc = sol_location(1);
            for j = 1:(size(find(fronts == val),1)-1)
                if d_2(sol_location(j+1),1) < best_loc
                    last_f_best = d_2(sol_location(j+1),1);
                    best_loc = sol_location(j+1);
                end
            end
            next_gen_pop(i,:) = merged_pop(best_loc,:);
       end
   end
  
   parent_pop = next_gen_pop;
   generation = generation + 1;
   
   %If dimension 2, plot objective space and decision space, respectively
   %if(dim == 2)
        clf;
        subplot(2,1,1);
        scatter(fitness(:,1),fitness(:,2));
        subplot(2,1,2);
        scatter(parent_pop(:,1), parent_pop(:,2));  
        drawnow
   %end
   
end






function rp_fronts = non_RPD_dominated_sorting(merged_pop, fitness, d_1, d_2, fitness_extremes)
    size_pop = size(merged_pop,1);
    num_obj = size(fitness,2);
    %p_dom = true(size_pop, size_pop);
    p_dom = false(size_pop, size_pop);
    total_p_dom = true(size_pop, size_pop);
    rp_dom = true(size_pop, size_pop);
    p_fronts = zeros(size_pop,1);
    rp_fronts = zeros(size_pop,1);
    fronts = zeros(size_pop,2);
    
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
    fronts = [p_fronts rp_fronts];
    
end


function sol = objective(m_pop, idx)   
    sol = zeros(size(m_pop,1),1);
        switch idx
            case 1
                %Auckleys
                %sol = m_pop(:,1) + m_pop(:,2); % + m_pop(:,3) + m_pop(:,4) + m_pop(:,5);
                %for i = 1:size(m_pop,1)
                %    term1_sum = sum(m_pop(i,:).^2);
                %    term2_sum = sum(cos(2*pi*m_pop(i,:)));
                %    sol(i,1) = -20*exp(-0.2*sqrt((1/2)*term1_sum)) - exp((1/2)*term2_sum) + 20 + exp(1);
                
                %Binh and Korn
                %sol = 4.*(m_pop(:,1).^2) + 4.*(m_pop(:,2).^2);
            
                %DTLZ 1, 2, 3
                sol = m_pop(:,1);
                
            case 2
                %High Conditioned E
                %for i = 1:size(m_pop,1)
                %    F1 = zeros(1,2);
                %    for a = 1:2
                %        F1(a) = (10^6)^((a-1)/(2-1));
                %    end
                %sol(i,1) = sum(F1.*m_pop(i,:).^2);
                
                %Binh and Korn
                %sol = (m_pop(:,1) - 5).^2 + (m_pop(:,2) - 5).^2;
                
                %DTLZ1
                %g = 1 + (9/29)*(sum(m_pop(:,2:end)'))';
                %h = 1 - sqrt(m_pop(:,1)./g);
                %sol = g.*h;
                
                %DTLZ3
                g = 1 + (9/29)*(sum(m_pop(:,2:end)'))';
                h = 1 - sqrt(m_pop(:,1)./g) - (m_pop(:,1)./g).*sin(10*pi*m_pop(:,1));
                sol = g.*h;
         end
              
        end






function d_1 = calc_d1(fit, ref, d_2)
    [pop_size, num_f] = size(fit);
    d_1 = zeros(pop_size,1);
    for n = 1:pop_size 
        d_1(n) = dot(fit(n,:), ref(d_2(n,2),:));
    end
end

function d_2 = calc_d2(fit, RPset)
    [pop_size, num_f] = size(fit);
    d_2 = zeros(pop_size,2);
    for n = 1:pop_size 
        d2_dist = sqrt(sum((fit(n,:)-RPset)'.^2)');
        [d_2(n,1), d_2(n,2)] = min(d2_dist);
    end
    
end

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
