data = dataset('File','input','Delimiter',' ','ReadVarNames',false);
data = double(data);
%datatest = dataset('File','S:\windows\project2\input_test','Delimiter',' ','ReadVarNames',false);
%datatest = double(datatest);

seq = dataset('File','seq','Delimiter',' ','ReadVarNames',false);
seq = double(seq);

D = numel(data(1,:));

%training data set
traindata = data(:,1:D-1);
traintarget = data(:,D:D);


numberoftraindata = numel(traindata(:,1));
bias = ones(numberoftraindata,1);
traindata = [bias traindata]; %add bias

%Validation data set
for v = 1:10
    if v==1
        validationdata = data(1:400,1:D-1);
        Tvalidationdata = data(1:400,D:D);
    else
        validationdata = vertcat(validationdata,data((v-1)*2000+1:(v-1)*2000+400,1:D-1));
        Tvalidationdata = vertcat(Tvalidationdata,data((v-1)*2000+1:(v-1)*2000+400,D:D));
    end
end

for m=1:numel(Tvalidationdata)
    tmp = Tvalidationdata(m);
    T = zeros(1,10);
    T(tmp+1)=1;
    if m==1
        Tvalidate = T;
    else
        Tvalidate = vertcat(Tvalidate,T);
    end
end

for m=1:numel(traintarget)
    tmp = traintarget(m);
    T = zeros(1,10);
    T(tmp+1)=1;
    if m==1
        Ttrain = T;
    else
        Ttrain = vertcat(Ttrain,T);
    end
end

numberofvalidatedata = numel(validationdata(:,1));
bias = ones(numberofvalidatedata,1);
validationdata = [bias validationdata]; %add bias
D = numel(traindata(1,:));
M = 352;
wupdate{1} = zeros(D,M);
wupdate{2} = zeros(M,10);
yita = 0.01;
%initialize w
for i=1:D
    for h=2:M
        w{1}(i,h) = rand(1)-0.5;
    end
end
for i = 1:M
    for h =1:10
        w{2}(i,h) = rand(1)-0.5;
    end
end

numberofpropagation=2000;

%start training and backpropagation
for numofprop = 1:numberofpropagation
    wupdate{1} = zeros(D,M);
    wupdate{2} = zeros(M,10);
    for sort = 1:numberoftraindata
        input = seq(sort);
    %for input = 1:1
    %    for hidden = 2:M 
    %        a(input,hidden) = 0;
    %        z(input,hidden) = 0;
    %        for id = 1:D
    %            a(input,hidden) = a(input,hidden) + traindata(input,id) * w(1,id,hidden);
    %        end
    %        z(input,hidden) = tanh(a(input,hidden)); %z(input,hidden) is the hideen unit value with activation function for input data input
    %        z(input,1) = 1;
    %    end
        
        a{input} = traindata(input,:) * w{1};
        z{input} = tanh(a{input});
        z{input}(1) = 1; %bias
        
        %for output = 1:10
        %    o(input,output) = 0;
        %    for hidden = 1:M
        %        o(input,output) = o(input,output) + w(2,hidden,output) * z(input,hidden); %this is the output without activation function
        %    end
        %end
        
        o{input} = z{input} * w{2};
        
        sumoutput = 0;
        for output = 1:10
            sumoutput = sumoutput + exp(o{input}(output));
        end
        
        %for output = 1:10
        %    oa(input,output) = exp(o(input,output))/sumoutput;%this is the output with the softmax function
        %end
        
        oa{input} = exp(o{input})/sumoutput;
        
        for u = 1:10
            if oa{input}(u) ==0
                oa{input}(u) = 10^-6;
            end
        end
        
        %oa is the output with softmax function
        %z  is the hidden unit with activation function
        %diet(2,input,output) is the diet at the output layer
        %diet(1,input,hidden) is the diet at the hidden unit layer

        %for output = 1:10
        %    diet(2,input,output) = oa{input}.*Ttrain(input,:) - Ttrain(input,:);
        %end
        
        diet{input}{2} = oa{input}.*Ttrain(input,:) - Ttrain(input,:);

        % this following code is used to calculate the derivative of the error
        % function for each layer
        %for hidden = 1:M
        %    for output = 1:10
        %        wupdate(2,hidden,output) = wupdate(2,hidden,output) + diet(2,input,output) * z(input,hidden);
        %    end
        %end
        
        wupdate{2} = wupdate{2} + z{input}' * diet{input}{2};

        %for hidden = 1:M
        %    diet(1,input,hidden) = 0;
        %    w2sum(hidden) = 0;
        %    for output = 1:10
        %        w2sum(hidden) = w2sum(hidden) + w(2,hidden,output) * diet(2,input,output);
        %    end
        %    diet(1,input,hidden) = (1-z(input,hidden)^2)*w2sum(hidden);
        %end
        
        diet{input}{1} = (1-z{input}.^2)' .* (w{2} * diet{input}{2}');

        %for i=1:D
        %    for hidden = 1:M
        %        wupdate(1,i,hidden) = wupdate(1,i,hidden) + diet(1,input,hidden) * traindata(input,i);
        %    end
        %end
        
        wupdate{1} = wupdate{1} + traindata(input,:)' * diet{input}{1}';
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf('propagation %d data %d seq %d OK\n',numofprop,sort,input);
        
        if numofprop > 1
             fprintf('EW %f\n',EW);
             fprintf('yita %f\n',yita);
        end
    end
    
        %the following code is used to update the weights
        %for i=1:D
        %    for hidden = 1:M
        %        w(1,i,hidden) = w(1,i,hidden) - yita * wupdate(1,i,hidden);
        %    end
        %end
        
        
        if numofprop == 1
            Eold = 10^8; 
        end
        EW = 0;
        for sort = 1:numberoftraindata
            input = seq(sort);
            for output =1:10
                EW = EW + Ttrain(input,output) * log(oa{input}(output));
            end
        end
        EW = -EW;

        if EW > Eold
           % numofprop = numofprop - 1;
           fprintf('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n');
            %EW = Eold;
            w{1} = wold1;
            w{2} = wold2;
            wupdate{1} = woldupdate1;
            wupdate{2} = woldupdate2;
            yita = yita/2;
            w{1} = w{1} - yita * wupdate{1};
            w{2} = w{2} - yita * wupdate{2};
            
            %w{1} = w{1}/100;
            %w{2} = w{2}/100;
            
            continue;
        end
        Eold = EW;
        E(numofprop) = EW;
        fprintf('propagation %d EW is: %d\n',numofprop,EW);


        wold1 = w{1};
        wold2 = w{2};
        woldupdate1 = wupdate{1};
        woldupdate2 = wupdate{2};

        w{1} = w{1} - yita * wupdate{1};
        
        %for hidden = 1:M
        %    for output = 1:10
        %        w(2,hidden,output) = w(2,hidden,output) - yita * wupdate(2,hidden,output);
        %    end
        %end
        
        w{2} = w{2} - yita * wupdate{2};
        
        
        %w{1} = w{1}/100;
        %w{2} = w{2}/100;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    %save data
    %fn = sprintf('w%d.mat',numofprop);
    %save(fn,'w','EW');
    %fn = sprintf('wupdate%d.mat',numofprop);
    %save(fn,'wupdate');
    %fn = sprintf('oa%d.mat',numofprop);
    %save(fn,'oa');
    %fn = sprintf('z%d.mat',numofprop);
    %save(fn,'z');
    %fn = sprintf('diet%d.mat',numofprop);
    %save(fn,'diet');
    if rem(numofprop,30) == 0
        fn = sprintf('w%d.mat',numofprop);
        save(fn,'w');
    end
end
