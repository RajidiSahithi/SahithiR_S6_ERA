# PART 1
## What is BackPropagation?
Backpropagation is an algorithm that backpropagates the errors from the output nodes to the input nodes. Therefore, it is simply referred to as the backward propagation of errors.

## Need for Backpropagation:
Backpropagation is “backpropagation of errors” and is very useful for training neural networks. It’s fast, easy to implement, and simple. Backpropagation does not require any parameters to be set, except the number of inputs. Backpropagation is a flexible method because no prior knowledge of the network is required.
Here are some of the advantages of the backpropagation algorithm:

* It’s memory-efficient in calculating the derivatives, as it uses less memory compared to other optimization algorithms, like the genetic algorithm. This is a very important feature, especially with large networks.
* The backpropagation algorithm is fast, especially for small and medium-sized networks. As more layers and neurons are added, it starts to get slower as more derivatives are calculated. 
* This algorithm is generic enough to work with different network architectures, like convolutional neural networks, generative adversarial networks, fully-connected networks, and more.
* There are no parameters to tune the backpropagation algorithm, so there’s less overhead. The only parameters in the process are related to the gradient descent algorithm, like learning rate.

### Example to understand Backpropagation (Feedforward network with one hidden layer and sigmoid loss)
Let us see the simple Neural Network shown below

![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/Simple_NN.png)  

This Neural Network is having a input layer, hidden layer and output layer.

This figure shows an example of a fully-connected artificial neural network (FCANN), the simplest type of network for demonstrating how the backpropagation algorithm works. The network has an input layer, 1 hidden layers, and an output layer. In the figure, the network architecture is presented horizontally so that each layer is represented vertically from left to right. 

Each layer consists of 1 or more neurons represented by circles. Because the network type is fully-connected, then each neuron in layer i is connected with all neurons in layer i+1.
In the above image 
<pre>
-i1,i2 are the inputs
-w1,w2,w3,w4 are weights of the input layer
-w5,w6,w7,w8 are weights from hidden layer 
-t1,t2 are desired or target values
</pre>
For each connection, there is an associated weight. The weight is a floating-point number that measures the importance of the connection between 2 neurons. The higher the weight, the more important the connection. The weights are the learnable parameter by which the network makes a prediction. If the weights are good, then the network makes accurate predictions with less error. Otherwise, the weight should be updated to reduce the error.
<br/> Assume that a neuron i1 at input layer  is connected to another neuron at hidden layer. Assume also that the value of h1 is calculated according to the next linear equation.
<pre>
h1,h2 are outputs of hidden layer. These are  the sum of products (SOP) between each input and its corresponding weight:
h1 = w1*i1 + w2*i2   
h2 = w3*i1 + w4*i2
</pre>
Each neuron in the hidden layer uses an activation function like sigmoid. The neurons in the output layer also use activation functions like sigmoid (for regression).
<pre>
-a_h1,a_h2 are sigmoid function (activation function) of h1,h2 respectively
a_h1 = σ(h1) = 1/(1 + exp(-h1))
a_h2 = σ(h2) = 1/(1 + exp(-h2))
-o1 = w5*a_h1 + w6*a_h2
-o2 = w7*a_h1 + w8*a_h2
-a_o1,a_o2 are are sigmoid function (activation function) of o1,o2 respectively
a_o1 = σ(o1) = 1/(1 + exp(-o1))
a_o2 = σ(o2) = 1/(1 + exp(-o2))
</pre>
To train a neural network, there are 2 passes (phases):
* Forward
* Backward
<br/>
In the forward pass, we start by propagating the data inputs to the input layer, go through the hidden layer(s), measure the network’s predictions from the output layer, and finally calculate the network error based on the predictions the network made. 
<br/> This network error measures how far the network is from making the correct prediction. For example, if the correct output is 0.5 and the network’s prediction is 0.3, then the absolute error of the network is 0.5-0.3=0.2. Note that the process of propagating the inputs from the input layer to the output layer is called forward propagation. Once the network error is calculated, then the forward propagation phase has ended, and backward pass starts.
<br/> The following formulas represent Errors with respect to targets.
<pre>
E1,E2 are the error with respect to target values t1,t2
E1 = ½ * (t1 - a_o1)²
E2 = ½ * (t2 - a_o2)²
-Total error E_Total = E1 + E2
</pre>
<br/>In the backward pass, the flow is reversed so that we start by propagating the error to the output layer until reaching the input layer passing through the hidden layer(s). The process of propagating the network error from the output layer to the input layer is called backward propagation, or simple backpropagation. The backpropagation algorithm is the set of steps used to update network weights to reduce the network error.

The forward and backward phases are repeated from some epochs. In each epoch, the following occurs:
* The inputs are propagated from the input to the output layer.
* The network error is calculated.
* The error is propagated from the output layer to the input layer.

Calculating gradients with the chain rule
Since a neural network has many layers, the derivative of C at a point in the middle of the network may be very far removed from the loss function, which is calculated after the last layer.
<br/> The output of the activation function from the output neuron reflects the predicted output of the sample. It’s obvious that there’s a difference between the desired and expected output.
<br/> Knowing that there’s an error, what should we do? We should minimize it. To minimize network error, we must change something in the network. Remember that the only parameters we can change are the weights and biases. We can try different weights and biases, and then test our network.

<br/> We calculate the error, then the forward pass ends, and we should start the backward pass to calculate the derivatives and update the parameters.

To practically feel the importance of the backpropagation algorithm, let’s try to update the parameters directly without using this algorithm.

<br/> To calculate the derivative of the error W.R.T the weights, simply multiply all the derivatives in the chain from the error to each weight,

#### Calculating Total Loss (E_Total) Gradient with respect to weights (w5,w6,w7,w8)
<pre>
∂E_total/∂w5 = ∂(E1 + E2)/∂w5
        E2 is independent of w5 so  ∂E_total/∂w5 = ∂E1/∂w5
        ∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5
        Let’s calculate partial derivatives of each part of the chain we created.
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
 #### Calculating Total Loss (E_Total) Gradient with respect to ----- (h1,h2)
 <pre> 
 ∂E_total/∂a_h1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) 
 Similarly
 ∂E_total/∂a_h2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8)
 </pre>
#### Calculating Total Loss (E_Total) Gradient with respect to weights (w1,w2,w3,w3) 
<pre>
∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1 
         we have
                ∂E_total/∂a_h1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) 
                ∂a_h1/∂h1 = a_h1 * (1 - a_h1) 
                ∂h1/∂w1 = i1
        Therefore ∂E_total/∂w1 =  ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
 
 ∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
 ∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
 ∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1
 ∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2
 </pre>  
##### Calculation of New Parameters
New parameters are calculated by using the formulas
<pre>
  New weight(w1) = Old weight(w1) - ƞ * ∂E_total/∂w1
  The same formula for remaining weights (w2,w3,w4,w5,w6,w7,w8)
</pre>
Based on the new parameters, we will recalculate the predicted output. The new predicted output is used to calculate the new network error. The network parameters are updated according to the calculated error. The process continues to update the parameters and recalculate the predicted output until it reaches an acceptable value for the error.
<br/> One important operation used in the backward pass is to calculate derivatives. Before getting into the calculations of derivatives in the backward pass, we can start with a simple example to make things easier.

 ###### Calculating the backpropagation for target values t1=0.5,t2=0.5,i1=0.05,i2=0.1,w1=0.15,w2=0.2,w3=0.25,w4=0.3,w5=0.4,w6=0.45,w7=0.5,w8=0.55 with learning rate = 1
 
 <br/> After calculating the individual derivatives in all chains, we can multiply all of them to calculate the desired derivatives (i.e. derivative of the error W.R.T each weight). we get the following values as mentioned in Screenshot below.
 
 ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/excel_screenshot.png)  
 
 Following is the graph from the above excel sheet
 
 ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr1.png)  
 
 <pre>             *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*# *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*#        </pre>
###### ERROR GRAPH WITH LEARNING RATE=0.1 
  ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr0.1.png)  
 <pre>             *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*# *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*#        </pre>
###### ERROR GRAPH WITH LEARNING RATE=0.2
  ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr0.2.png) 
 <pre>             *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*# *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*#        </pre>
###### ERROR GRAPH WITH LEARNING RATE=0.5
  ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr0.5.png)  
 <pre>             *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*# *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*#        </pre>
###### ERROR GRAPH WITH LEARNING RATE=1.0
  ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr1.0.png)   
 <pre>             *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*# *#*#*#*# *#*#*#*#*#*#*#*# *#*#*#*#*#*#*#*#       </pre>
###### ERROR GRAPH WITH LEARNING RATE=2.0
  ![alt text](https://github.com/RajidiSahithi/SahithiR_S6_ERA/blob/main/Images/lr2.0.png)   

# PART 2

 


