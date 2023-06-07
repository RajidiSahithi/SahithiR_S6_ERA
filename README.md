# PART 1
## What is BackPropagation?
Backpropagation is an algorithm that backpropagates the errors from the output nodes to the input nodes. Therefore, it is simply referred to as the backward propagation of errors.

## Need for Backpropagation:
Backpropagation is “backpropagation of errors” and is very useful for training neural networks. It’s fast, easy to implement, and simple. Backpropagation does not require any parameters to be set, except the number of inputs. Backpropagation is a flexible method because no prior knowledge of the network is required.

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
-a_h1,a_h2 are sigmoid function (activation function) of h1,h2 respectively
a_h1 = σ(h1) = 1/(1 + exp(-h1))
a_h2 = σ(h2) = 1/(1 + exp(-h2))
-o1 = w5*a_h1 + w6*a_h2
-o2 = w7*a_h1 + w8*a_h2
-a_o1,a_o2 are are sigmoid function (activation function) of o1,o2 respectively
a_o1 = σ(o1) = 1/(1 + exp(-o1))
a_o2 = σ(o2) = 1/(1 + exp(-o2))
E1,E2 are the error with respect to target values t1,t2
E1 = ½ * (t1 - a_o1)²
E2 = ½ * (t2 - a_o2)²
-Total error E_Total = E1 + E2
</pre>
Now lets see how back propagation can be done
#### Calculating Total Loss (E_Total) Gradient with respect to weights (w5,w6,w7,w8)
<pre>
∂E_total/∂w5 = ∂(E1 + E2)/∂w5
        E2 is independent of w5 so  ∂E_total/∂w5 = ∂E1/∂w5
        ∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5
            we have
                ∂E1/∂a_o1 =  ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1)
                ∂a_o1/∂o1 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)
                ∂o1/∂w5 = a_h1
        Therefore ∂E1/∂w5 =  (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1
 
 ∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1 
 ∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2
 ∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1
 ∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2
 </pre>
#### Calculating Total Loss (E_Total) Gradient with respect to weights (w1,w2,w3,w3) 
<pre>
∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1 
   
        ∂E2/∂w1 = ∂E2/∂a_o2 * ∂a_o2/∂o2 * ∂o1/∂w1
            we have
                ∂E_total/∂a_h1 =  
                ∂a_h1/∂h1 =  
                ∂h1/∂w1 = a_h2
        Therefore ∂E_total/∂w1 =  (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2
 
 ∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1 
 ∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2
 ∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1
 ∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2
 </pre>              


