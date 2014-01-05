data = dataset('File','input','Delimiter',' ','ReadVarNames',false);
data = double(data);

seq = dataset('File','seq','Delimiter',' ','ReadVarNames',false);
seq = double(seq);

D = numel(data(1,:));

%training data set
traindata = data(:,1:D-1);
traintarget = data(:,D:D);

numberoftraindata = numel(traindata(:,1));
bias = ones(numberoftraindata,1);
traindata = [bias traindata]; %add bias

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

dimension = numel(traindata(1,:));
outputdimension = 10;

%initial w
%w = rand(dimension,outputdimension)-0.5;
w = ones(dimension,outputdimension);
yita = 0.001;
interations = 3000;

for i = 1:interations
    gradient = zeros(dimension,outputdimension);
    for sort = 1:numberoftraindata
        input = seq(sort);
        a{input} = traindata(input,:) * w;

        sum = 0;
        for output = 1:outputdimension
            sum = sum + exp(a{input}(output));
        end
        z{input} = exp(a{input}) / sum;

        for u = 1:10
            if z{input}(u) == 0
                z{input}(u) = 10^-6;
            end
        end
        
        %z is the output with softmax function

        diet{input} = z{input} - Ttrain(input,:);
        gradient = gradient + traindata(input,:)' * diet{input};
        
        fprintf('propagation %d data %d seq %d OK\n',i,sort,input);
        
        if i > 1
            fprintf('EW %f\n',EW);
            fprintf('yita %f\n',yita);
        end
    end
    
    EW = 0;
    for sort = 1:numberoftraindata
        input = seq(sort);
        for output =1:10
            EW = EW + Ttrain(input,output) * log(z{input}(output));
        end
    end
    EW = -EW;
    
    if i == 1
        Eold = 10^8; 
    end
    if EW > Eold
       % numofprop = numofprop - 1;
        %EW = Eold;
        w = wold;
        gradient = gradientold;
        yita = yita/2;
        w = w - yita * gradient;
        continue;
    end
    Eold = EW;
    E(i) = EW;
    
    fprintf('EW is: %f\n',EW);
    
    wold = w;
    gradientold = gradient;
        
    w = w - yita * gradient;
    
    %if rem(i,30) == 0
    %    fn = sprintf('S:\\windows\\project2\\lr\\wlr%d.mat',i);
    %    save(fn,'w');
    %end
end