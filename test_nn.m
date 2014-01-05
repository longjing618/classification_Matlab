datatest = dataset('File','S:\windows\project2\input_test','Delimiter',' ','ReadVarNames',false);
datatest = double(datatest);

D = numel(datatest(1,:));

%test data set
testdata = datatest(:,1:D-1);
testtarget = datatest(:,D:D);

numberoftestdata = numel(testdata(:,1));
bias = ones(numberoftestdata,1);
testdata = [bias testdata]; %add bias


for m=1:numel(testtarget)
    tmp = testtarget(m);
    T = zeros(1,10);
    T(tmp+1)=1;
    if m==1
        Ttest = T;
    else
        Ttest = vertcat(Ttest,T);
    end
end

%get class for test data set
for input = 1:numberoftestdata
    a{input} = testdata(input,:) * w{1};
    z{input} = tanh(a{input});
    z{input}(1) = 1; %bias

    o{input} = z{input} * w{2};

    sumoutput = 0;
    for output = 1:10
        sumoutput = sumoutput + exp(o{input}(output));
    end
    oa{input} = exp(o{input})/sumoutput;

    tmp = 1;
    for c = 1:10
        if(oa{input}(tmp) < oa{input}(c))
            tmp = c;
        end
    end 

    class(input) = tmp-1;
end

%calculate error rate
correct = 0;
for i = 1:numberoftestdata
    if class(i) == testtarget(i)
        correct = correct + 1;
    end
end

rate = 1 - correct / numberoftestdata;
%fprintf('The error rate is: %f\n',rate);
disp(class');