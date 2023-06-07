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

![alt text]()  #add link of image simple_NN

This Neural Network is having a input layer, hidden layer and output layer.
This figure shows an example of a fully-connected artificial neural network (FCANN), the simplest type of network for demonstrating how the backpropagation algorithm works. The network has an input layer, 2 hidden layers, and an output layer. In the figure, the network architecture is presented horizontally so that each layer is represented vertically from left to right. 

Each layer consists of 1 or more neurons represented by circles. Because the network type is fully-connected, then each neuron in layer i is connected with all neurons in layer i+1.
In the above images 
<pre>
-i1,i2 are the inputs
-w1,w2,w3,w4 are weights of the input layer
-w5,w6,w7,w8 are weights from hidden layer 
-t1,t2 are desired or target values
h1,h2 are output of hidden layer. These are  the sum of products (SOP) between each input and its corresponding weight:
h1 = w1*i1 + w2*i2   
h2 = w3*i1 + w4*i2
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
Calculating gradients with the chain rule
Since a neural network has many layers, the derivative of C at a point in the middle of the network may be very far removed from the loss function, which is calculated after the last layer.
In fact, C depends on the weight values via a chain of many functions. We can use the chain rule of calculus to calculate its derivate. The chain rule tells us that for a function z depending on y, where y depends on x, the derivate of z with respect to x is given by:
<pre>
                                                        ∂z/∂x = ∂z/∂y * ∂y/∂x

</pre>

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
 
 ###### Calculating the backpropagation for target values t1=0.5,t2=0.5,i1=0.05,i2=0.1,w1=0.15,w2=0.2,w3=0.25,w4=0.3,w5=0.4,w6=0.45,w7=0.5,w8=0.55 with learning rate = 1
 
 ![alt text]()  #add link of excel_screenshot
 
 ![alt text]()  #add link of lr1
 
 
 
###### ERROR GRAPH WITH LEARNING RATE=0.1 
  ![alt text]()  #add link of lr0.1
###### ERROR GRAPH WITH LEARNING RATE=0.2
  ![alt text]()  #add link of lr0.2
###### ERROR GRAPH WITH LEARNING RATE=0.5
  ![alt text]()  #add link of lr0.5
###### ERROR GRAPH WITH LEARNING RATE=1.0
  ![alt text]()  #add link of lr1.0 
###### ERROR GRAPH WITH LEARNING RATE=2.0
  ![alt text]()  #add link of lr2.0  

             
 


