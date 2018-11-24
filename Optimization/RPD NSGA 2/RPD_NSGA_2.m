% RPD-NSGA-II

pop_size = 100; %Population Size
num_obj = 2; %Number of objectives
num_div = 20; %Number of divisions
t = 0;
dim = 5;
global zeta_c;
global zeta_m;
zeta_c = 20;
zeta_m = 20;
p_c = 0.9; %Probability of crossover
p_m = 0.05; %Probability of mutation
generation = 0;

range = ones(dim, 2); %Create range of variables for evaluation in
%the form of [max min; max min; ...]

range(:,1) = range(:,1)*1000; %Max is 1000
range(:,2) = range(:,2)*-1000; %Min is -1000

%Initialize population
parent_pop = zeros(pop_size,dim);
variation_pop = zeros(pop_size,dim);


parent_pop = rand(pop_size,dim);
for i = 1:dim
    %Map random numbers to variable range = rand*(max - min) + min
    parent_pop(:,i) =  parent_pop(:,i).*(range(i,1) - range(i,2)) +  range(i,2);
end 

%Reference Point Set
% Based on normalized hyperplane
num_combs = nchoosek((num_obj + num_div - 1), num_div);
normalized_vec = linspace(0,1, num_div + 1);  
RPset = zeros(num_combs, num_obj);

A = permn(normalized_vec, num_obj);
B = sum(A') == 1;
C = find(B);

%Generate Reference Points
for i = 1:sum(B)
    RPset(i,:) = A(C(i),:);
end

%Line 5
while generation < 1000 
   variation_pop = variation(parent_pop, range); 
   merged_pop = union(parent_pop, variation_pop, 'rows');
   
   generation = generation + 1; 
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
