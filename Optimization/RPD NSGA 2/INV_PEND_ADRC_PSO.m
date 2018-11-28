
M = 0.500; %190g for linear bearing*2 + 11.4g*4 bearings + 60g plastic + 50g metal rod
m = 1.5; %Approximately 800grams
b = 0.5;
l = 0.3;
I = 0.006;
g = 9.81;
d_p = 0.0127;
b_c = 0.2;
b_m = 0.2;
K_T = 0.5; %N*m/A
K_E = 0.5; %V*s/rad
R_a = 0.5; %Ohms 
J_m = 0.002;

h = 0.001; %Sampling period
global h_1;
global r;
global c;
global beta_01;
global beta_02;
global beta_03;
global b_0;

h_1 = 100;
r = 5;
c = 10;
beta_01 = 10;
%beta_02 = 1/(2*h^0.5); %5
%beta_02 = 1/(3*h);
beta_02 = 10;
%beta_03 = 2/(25*h^1.2); %20
%beta_03 = 2/(64*h^2);
beta_03 = 100;
beta_1 = 1;
beta_2 = 1;
b_0 = 10000;

sim('ADRC_Observer', 'CaptureErrors', 'on');


a_24u = (d_p/(2*m*l))*(b_c - (((M+m)*(b_m +(K_T*K_E)/(R_a)))/J_m));
a_41u = (2*g)/d_p;
a_44u = ((J_p + m*l)/(m^2*l^2))*(b_c - (((M+m)*(b_m +(K_T*K_E)/(R_a)))/J_m));
b_2u = (d_p*(M+m)*K_T)/(2*m*l*R_a); 
b_4u = ((J_p + m*l)*(M+m)*K_T)/(m^2*l^2*R_a);


%a_22 = -((I+m*l^2)*b)/(I*(M+m)+M*m*l^2);
%a_23 = (m^2*g*l^2)/(I*(M+m)+M*m*l^2);
%a_42 = -(m*l*b)/(I*(M+m)+M*m*l^2);
%a_43 = (m*g*l*(M+m))/(I*(M+m)+M*m*l^2);

%b_21 = (I+m*l^2)/(I*(M+m)+M*m*l^2);
%b_41 = (m*l)/(I*(M+m)+M*m*l^2);

%State Space Form

%A = [0 1 0 0; 0 a_22 a_23 0; 0 0 0 1; 0 a_42 a_43 0];
A_updated = [0 1 0 0; 0 0 0 a_24u; 0 0 0 1; a_41u 0 0 a_44u];
%B = [0; b_21; 0; b_41];
B_updated = [0; b_2u; 0; b_4u];
%C = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
C = [1 0 0 0];
%D = [0;0;0;0];





%fileID = fopen('HGBat_PSO_5.txt','w');
%fprintf(fileID,'%-13s %-13s %-13s %-13s\n','mean','worst','best','std');
pop_size = 100;
D = 7;
NFC = 0;
Max_NFC = 2000;
x_max = 1000;
x_min = 0;
max_runs = 5;

best_so_far = zeros(max_generations,Max_NFC);
best_params = zeros(max_generations,D);
C_1 = 2.05;
C_2 = 2.05;
v = zeros(pop_size,D); %Velocities
pi = zeros(pop_size, D); %personal best coords
pg = zeros(1,D); %Global Best coords
fi = 9999999*ones(pop_size,1); %Best local fitness (set high for min func)
fg = 9999999;% Best global fitness (set high for min func)
Wi = linspace(0.9, 0.4, Max_NFC); %Weight matrix
%Init pop
x = zeros(pop_size, D); %Population
for generation = 1:max_runs
    for i = 1:pop_size
        x(i,:) =  x_min + (x_max - x_min)*rand(D,1)';
    end
%While
    NFC = 0;
    best = 99999999;
    v = zeros(pop_size,D); %Velocities
    pi = zeros(pop_size, D); %personal best coords
    pg = zeros(1,D); %Global Best coords
    fi = 9999999*ones(pop_size,1); %Best local fitness (set high for min func)
    fg = 9999999;% Best global fitness (set high for min func)
    
    while(NFC < Max_NFC)
        for i = 1:pop_size
            [fitness(i,1), NFC] = fitness_(x(i,:), NFC) %Eval fitness
            generation
            v(i,:) = Wi(NFC)*v(i,:) + rand*C_1*(pi(i,:)-x(i,:)) + rand*C_2*(pg - x(i,:));
            if((max(x(i,:)) <= x_max) && (min(x(i,:)) >= x_min))
                if( fitness(i,1) <  fi(i,1) ) %if best local
                    pi(i,:) = x(i,:); %Update local best
                    fi(i,1) = fitness(i,1); %Update local best pos
                elseif( fitness(i,1) <  fg ) %if best global
                    pg = x(i,:); %Update global best
                    fg = fitness(i,1); %Update global best pos
                end
            else
                fitness(i,1) = 9999999;
            end
            if(fitness(i,1) < best)
                best = fitness(i,1);
                best_params(generation,:) = x(i,:);
            end
            best_so_far(generation, NFC) = best;
            x(i,:) = x(i,:) + v(i,:); %Move to next pos
        end
        if(D == 2)
            plot_result(x, x_max, x_min) %plot in 2D
        end
    end
end

[best_value, idx_best] = min(best_so_far(:,end));
best_run = best_so_far(idx_best,:);

function [fit, NFC] = fitness_(x,nfc)
    global h_1;
    global r;
    global c;
    global beta_01;
    global beta_02;
    global beta_03;
    global b_0;
    %sim_results = ones(5001,5)*30;
    h_1 = x(1,1);
    r = x(1,2);
    c = x(1,3);
    beta_01 = x(1,4);
    beta_02 = x(1,5);
    beta_03 = x(1,6);
    b_0 = x(1,7);
    res = sim('ADRC_Observer', 'CaptureErrors', 'on');
    
    fit = sum(abs(res.sim_results(:,4)));
    NFC = nfc + 1;
end





