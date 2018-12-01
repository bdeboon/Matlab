%PSO
fileID = fopen('HGBat_PSO_5.txt','w');
fprintf(fileID,'%-13s %-13s %-13s %-13s\n','mean','worst','best','std');
pop_size = 100;
D = 5;
NFC = 0;
Max_NFC = 5000*D;
x_max = 10;
x_min = -10;
num_runs = 31;
%PSO Init
best_so_far = zeros(num_runs,Max_NFC);

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
for runs = 1:num_runs
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
            [fitness(i,1), NFC] = fitness_(x(i,:), NFC); %Eval fitness
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
            end
            best_so_far(runs, NFC) = best;
            x(i,:) = x(i,:) + v(i,:); %Move to next pos
        end
        if(D == 2)
            plot_result(x, x_max, x_min) %plot in 2D
        end
    end
end

%Save to txt
mean_val = mean(best_so_far);
std_val = std(best_so_far);
[best_value, idx_best] = min(best_so_far(:,end));
[worst_value, idx_worst] = max(best_so_far(:,end));
best_run = best_so_far(idx_best,:);
worst_run = best_so_far(idx_worst,:);
NFC_saved = linspace(1, Max_NFC, Max_NFC);
t = 1:Max_NFC;
A = [mean_val; worst_run; best_run; std_val];
fprintf(fileID,'%u %u %u %u\n',A);
fclose(fileID);

%%%%%%%%%%
clf;
plot(t, mean_val)
hold on;
plot(t, worst_run)
hold on;
plot(t, best_run)
hold on;
plot(t, std_val)
hold off;
legend('Mean','Worst','Best','Std Dev.');

%1.) High Conditioned Elliptic Function
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    F1 = zeros(1,D);
    for a = 1:D
        F1(a) = (10^6)^((a-1)/(D-1));
    end
    fit = sum(F1.*x(1,:).^2);
    NFC = nfc + 1;
end
%}

%2.) Bent Cigar
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    F2 = zeros(1,D);
    F2(1) = 1;
    for a = 2:D
        F2(a) = (10^6);
    end
    fit = sum(F2.*x(1,:).^2);
    NFC = nfc + 1;
end
%}

%3.) Discus Function
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    F3 = zeros(1,D);
    F3(1) = 10^6;
    for a = 2:D
        F3(a) = 1;
    end
    fit = sum(F3.*x(1,:).^2);
    NFC = nfc + 1;
end
%}

%4.) Rosenbrock's
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    term1 = zeros(1,D);
    term1 = (100*(x(1,1:D-1).^2-x(1,2:D)).^2 + (x(1,1:D-1)-1).^2);
    fit = sum(term1);
    NFC = nfc + 1;
end
%}

%5.)Auckleys
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    term1_sum = sum(x(1,:).^2);
    term2_sum = sum(cos(2*pi*x(1,:)));
    fit = -20*exp(-0.2*sqrt((1/D)*term1_sum)) - exp((1/D)*term2_sum) + 20 + exp(1);    
    NFC = nfc + 1;
end
%}

%6 Weierstrass Function
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    a = 0.5;
    b = 3;
    k_max = 20;
    term1 = zeros(k_max+1,D);
    term2 = zeros(1, k_max+1);
    D_mult = 1:D;
    for k=0:k_max       
        term1(k+1,:) = (a^k*cos(2*pi*b^k*(x(1,:) + 0.5)));
        term2(k+1) = a^k*cos(2*pi*b^k*0.5);
    end
    fit = sum(sum(term1)-D_mult*sum(term2));
    NFC = nfc + 1;
end
%}

%7 Griewank's Function
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    D_mult = 1:D;
    term1 = x(1,:).^2/4000;
    term2 = cos(x(1,:)./D_mult);
    fit = sum(term1)-prod(term2)+1;
    NFC = nfc + 1;
end
%}

%8 Rastrigin's Function
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    term1 = zeros(1,D);
    term1 = x(1,:).^2-10*cos(2*pi*x(1,:))+10;
    fit = sum(term1);
    NFC = nfc + 1;
end
%}

%9.) Modified Schwefel's Function (for x ranging -10,10)
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    z = x(1,:) + 420.9687462275036;
    term1 = z.*sin(abs(z).^0.5);
    fit = 418.9829*D - sum(term1);
    NFC = nfc + 1;
end
%}


%10.) Katsuura
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    
    term1_sum = zeros(1,D);
    term2 = zeros(1,D);
    prod_term = 1;
    for i = 1:D
        term1_sum = zeros(1,D);
        for j = 1:32
            term1_sum(i) = term1_sum(i) + abs((2^j.*x(1,i) - round(2^j.*x(1,i)))/2^j);
        end
        term1_sum = sum(term1_sum);
        term2(i) = ((1+i*term1_sum)^(10/D^1.2));
        prod_term = prod_term*term2(i);
    end
    fit = ((10/D^2)*prod_term)-(10/D^2);
    NFC = nfc + 1;
end
%}

%11.) Happy Cat
%{
function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    term1 = x(1,:).^2;
    fit = abs(sum(term1)-D)^0.25 + (0.5*sum(term1)+sum(x(1,:)))/D + 0.5;
    NFC = nfc + 1;
end
%}

%12 HGBat Function

function [fit, NFC] = fitness_(x,nfc)
    D = size(x,2);
    term1 = x(1,:).^2;
    term2 = abs(sum(term1)^2-(sum(x(1,:)))^2)^0.5;
    term3 = (0.5*sum(term1)+sum(x(1,:)))/D;
    fit = term2 + term3 +0.5;
    NFC = nfc + 1;
end


function plot_result(x, x_max, x_min)    
    clf;
    axis([x_min x_max x_min x_max]);
    hold off;
    plot(x(:,2)',x(:,1)','o')
    axis([x_min x_max x_min x_max]);
    drawnow
end

