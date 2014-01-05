datatest = dataset('File','S:\windows\project2\input_test','Delimiter',' ','ReadVarNames',false);
datatest = double(datatest);

D = numel(datatest(1,:));
outputdimension = 10;

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
    a{input} = testdata(input,:) * w;

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

    tmp = 1;
    for c = 1:10
        if(z{input}(tmp) < z{input}(c))
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