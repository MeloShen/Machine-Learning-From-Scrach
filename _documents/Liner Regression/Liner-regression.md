# Implement the regression of linear function

[Code](https://github.com/MeloShen/Machine-Learning-From-Scrach/blob/main/_documents/Liner%20Regression/regression.py）

## Confirm training data
| x     | y     | x     | y     | x     | y     | x     | y     |
|-------|-------|-------|-------|-------|-------|-------|-------|
|235    |591    |216	|539    |59	    |319    |198    |522    |
|148	|413    |35	    |310    |159    |400    |159    |427    |
|85	    |308    |204	|519    |162    |425    |272    |659    |
|49	    |325    |25	    |332    |117    |385    |112    |37     |
|173    |498    |191    |498    |134    |392    |99	    |334    |

[click.csv](https://github.com/MeloShen/Machine-Learning-From-Scrach/blob/main/_documents/Liner%20Regression/click.csv)

## Read the traning data draw the Matplotlib
![regress_image_1.jpg](https://raw.githubusercontent.com/MeloShen/Machine-Learning-From-Scrach/main/_documents/Liner%20Regression/_image/regress_image_1.jpg)

## Implemented as a one-time function

$f_{\theta}(x) = \theta_{1} + \theta_{2}x$

$E(\theta )=\frac{1}{2} \sum_{i=1}^{n} \left ( y^{(i)} - f_{\theta}(x^{(i)})\right )^{2}$

Initialize the $\theta_{1}$ and $\theta_{2}$ with a random number, using code to express prediction and objective functions in python.After that, the training data should be changed into data with an average value of 0 and a variance of 1, so that the convergence of parameters can be faster $z^{(i)} =  \frac{x^{(i)} - \mu }{\delta }$ .

![regress_image_2.jpg](https://raw.githubusercontent.com/MeloShen/Machine-Learning-From-Scrach/main/_documents/Liner%20Regression/_image/regress_image_2.jpg)

The scale on the y axis shrinks immediately.

## Update parameter



$\theta_{0}:\theta_{0}-\eta\sum_{i=1}^{n}(f_{\theta}(x^{(i)})-y^{(i)})$

$\theta_{1}:\theta_{1}-\eta\sum_{i=1}^{n}(f_{\theta}(x^{(i)})-y^{(i)})$

For $\eta$, we need to keep trying to get the definite value. After running for many times, we can compare the value of the target function. If there is no change, we can finish learning. And when the parameter is updated, the same iteration value must be used for $\theta _{0}$and $\theta _{0}$.

### The log
[log.txt](https://github.com/MeloShen/Machine-Learning-From-Scrach/tree/main/_documents/Liner%20Regression)
It is 469: theta0 = 429.117,theta1 = 135.643,difference = 0.0134

It is 470: theta0 = 429.118,theta1 = 135.659,difference = 0.0132

It is 471: theta0 = 429.118,theta1 = 135.675,difference = 0.0129

It is 472: theta0 = 429.119,theta1 = 135.691,difference = 0.0126

It is 473: theta0 = 429.120,theta1 = 135.707,difference = 0.0124

It is 474: theta0 = 429.120,theta1 = 135.723,difference = 0.0121

It is 475: theta0 = 429.121,theta1 = 135.738,difference = 0.0119

It is 476: theta0 = 429.121,theta1 = 135.753,difference = 0.0117

It is 477: theta0 = 429.122,theta1 = 135.769,difference = 0.0114

It is 478: theta0 = 429.123,theta1 = 135.784,difference = 0.0112

It is 479: theta0 = 429.123,theta1 = 135.798,difference = 0.0110

It is 480: theta0 = 429.124,theta1 = 135.813,difference = 0.0107

It is 481: theta0 = 429.124,theta1 = 135.828,difference = 0.0105

It is 482: theta0 = 429.125,theta1 = 135.842,difference = 0.0103

It is 483: theta0 = 429.125,theta1 = 135.857,difference = 0.0101

It is 484: theta0 = 429.126,theta1 = 135.871,difference = 0.0099

From the log, the difference is already small, but because the initial value of $\theta$ is random, the number of loops and the reduction in difference will vary from execution to execution.

![egress_image_3.jpg](https://raw.githubusercontent.com/MeloShen/Machine-Learning-From-Scrach/main/_documents/Liner%20Regression/_image/regress_image_3.jpg)