# PART 1
## What is BackPropagation?
Backpropagation is an algorithm that backpropagates the errors from the output nodes to the input nodes. Therefore, it is simply referred to as the backward propagation of errors.

### Example to understand Backpropagation
Let us see the simple Neural Network shown below
![alt text]()  #add link of image simple_NN

This Neural Network is having a input layer, hidden layer and output layer.
In the above images 
<pre>
-i1,i2 are the inputs
-w1,w2,w3,w4 are weights from input layer to hidden layer
-w5,w6,w7,w8 are weights from hidden layer to output layer
-t1,t2 are desired or target values
-h1 = w1*i1 + w2*i2   
-h2 = w3*i1 + w4*i2
-a_h1,a_h2 are sigmoid functions of h1,h2 respectively
a_h1 = σ(h1) = 1/(1 + exp(-h1))
a_h2 = σ(h2) = 1/(1 + exp(-h2))
-o1 = w5*a_h1 + w6*a_h2
-o2 = w7*a_h1 + w8*a_h2
-a_o1,a_o2 are are sigmoid functions ofo1,o2 respectively
a_o1 = σ(o1) = 1/(1 + exp(-o1))
a_o2 = σ(o2) = 1/(1 + exp(-o2))
</pre>


Now lets see how back propagation can be done

