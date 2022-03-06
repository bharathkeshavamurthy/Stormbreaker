clear;clc

r{1}=csvread("radio0/send")-1.500243e9;
r{2}=csvread("radio1/send")-1.500243e9;
r{3}=csvread("radio2/send")-1.500243e9;
r{4}=csvread("radio3/send")-1.500243e9;
r{5}=csvread("radio4/send")-1.500243e9;

n = ones(1,5);

s = 0;
e = 0;

while 1
    ns = inf;
    nr = 0;
    
    for i = 1:5
        ls = r{i}(n(i),1);
        if ls < ns
            nr = i;
            ns = ls;
        end
    end
    
    ne = r{nr}(n(nr),2);
    
    n(nr) = n(nr)+1;
    
    assert(ns >= e);
    
    s = ns;
    e = ne;
end