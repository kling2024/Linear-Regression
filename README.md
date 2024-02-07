# Linear Regression
I implemented linear regression with one variable to predict profits for a restaurant franchise in this lab. (CPSC-5616EL-03 Machine Learning/Deep Learning)

## Problem Statement

Suppose I am the CEO of a restaurant franchise and are considering different cities for opening a new outlet.

- I would like to expand my business to cities that may give my restaurant higher profits.
- The chain already have restaurants in various cities and I have data for profits and populations from the cities.
- I also have data on cities that were candidates for a new restaurant. 
	- For these cities, I have the city population.
    
I would use the data to help me identify which cities may potentially give my business higher profits.

## Importing Packages 
I imported all the packages that I needed during this assignment, including numpy, matplotlib, etc.

## Loading Dataset

I started by loading the dataset named `ex1data1.txt` for this task. There were two columns in the dataset

- First column: `x_train` is the population of a city
- Second column: `y_train` is the profit of a restaurant in that city. A negative value for profit indicates a loss. 

For this dataset, I used a scatter plot to visualize the data, since it had only two properties to plot (profit and population).

My goal was to build a linear regression model to fit this data.

- With this model, I could then input a new city's population, and had the model to estimate my restaurant's potential monthly profits for that city.

## Creating Linear Regression Model

I fitted the linear regression parameters $(w,b)$ to my dataset.

- The model function for linear regression, which is a function that maps from `x` (city population) to `y` (restaurant's monthly profit for that city) is represented as 
    $$f_{w,b}(x) = wx + b$$
    

- To train a linear regression model, I want to find the best $(w,b)$ parameters that fit my dataset.  

    - To compare how one choice of $(w,b)$ is better or worse than another choice, I can evaluate it with a cost function $J(w,b)$
      
    - The choice of $(w,b)$ that fit my data the best is the one that had the smallest cost $J(w,b)$.


- To find the values $(w,b)$ that get the smallest possible cost $J(w,b)$, I can use a method called **gradient descent**. 
  - With each step of gradient descent, my parameters $(w,b)$ come closer to the optimal values that will achieve the lowest cost $J(w,b)$.
  

- The trained linear regression model can then take the input feature $x$ (city population) and output a prediction $f_{w,b}(x)$ (predicted monthly profit for a restaurant in that city).

## Computing Cost

Gradient descent involves repeated steps to adjust the value of my parameter $(w,b)$ to gradually get a smaller and smaller cost $J(w,b)$.

For one variable, the cost function for linear regression $J(w,b)$ is defined as

$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$ 

- $f_{w,b}(x^{(i)})$ is the model's prediction of my restaurant's profit, as opposed to $y^{(i)}$, which is the actual profit that is recorded in the data.
- $m$ is the number of training examples in the dataset

For linear regression with one variable, the prediction of the model $f_{w,b}$ for an example $x^{(i)}$ is representented as:

$$ f_{w,b}(x^{(i)}) = wx^{(i)} + b$$

This is the equation for a line, with an intercept $b$ and a slope $w$

Here, I implemented a function named `compute_cost()`to calculate the cost $J(w,b)$ so that I could check the progress of my gradient descent implementation

## Gradient descent 

The gradient descent algorithm is:

$$\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \ & w := w -  \alpha \frac{\partial J(w,b)}{\partial w} \newline       \& b := b -  \alpha \frac{\partial J(w,b)}{\partial b}; & 
\newline & \rbrace\end{align*}$$

Where, parameters $w, b$ are both updated simultaniously and where  

$$
\frac{\partial J(w,b)}{\partial w}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) -y^{(i)})x^{(i)}
$$

$$
\frac{\partial J(w,b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})
$$

Here, $m$ is the number of training examples and $\sum$ is the summation operator

I implemented a function called `compute_gradient` which calculates $\frac{\partial J(w)}{\partial w}$, $\frac{\partial J(w)}{\partial b}$ for linear regression. 

## Learning parameters using batch gradient descent 

I found the optimal parameters of this linear regression model by using batch gradient descent algorithm for my dataset

- A good way to verify that gradient descent is working correctly is to look
at the value of $J(w,b)$ and check that it is decreasing with each step. 

- After I implemented the gradient and computed the cost correctly, with appropriate value for the learning rate alpha, $J(w,b)$ never increased and converged to a steady value by the end of the algorithm.

**w, b found by gradient descent: 1.16636235, -3.63029143940436**

Then I used the final parameters from gradient descent to plot the linear fit.  

To calculate the predictions on the entire dataset, I looped through all the training examples and calculated the prediction for each example. Then, I plotted the predicted values to see the linear fit.

The final values of $w,b$ can also be used to make predictions on profits. For example, to predict what the profit would be in areas of 35,000 and 70,000 people. 

**For population = 35,000, we predict a profit of $4519.77**

**For population = 70,000, we predict a profit of $45342.45**







