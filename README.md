# csc413-2516-homework-2-solved
**TO GET THIS SOLUTION VISIT:** [CSC413-2516 Homework 2 Solved](https://www.ankitcodinghub.com/product/csc413-2516-homework-2-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;118061&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSC413-2516  Homework 2 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
(Q1.2) Added initial condition clarification.

(Q2.1.2) Reweighted to be 0.5 points

(Q2.3.1) Reweighted to be 0.5 points

(Q2.3) Change the weight decay equation to use X instead of xi

Submission: You must submit your solutions as a PDF file through MarkUs . You can produce the file however you like (e.g. LaTeX, Microsoft Word, scanner), as long as it is readable.

1 Optimization

We will continue using the linear regression model established in Homework 1. Given n pairs of input data with d features and scalar labels (xi,ti) ∈ Rd ×R, we want to find a linear model f(x) = wˆ T x with wˆ ∈Rd such that the squared error on training data is minimized. Given a data matrix X ∈Rn×d and corresponding labels t ∈Rn, the objective function is defined as:

(1)

1.1 Mini-Batch Stochastic Gradient Descent (SGD)

Mini-batch SGD performs optimization by taking the average gradient over a mini-batch, denoted

B ∈ Rb×d, where 1 &lt; b ≪ n. Each training example in the mini-batch, denoted xj ∈ B, is randomly sampled without replacement from the data matrix X. Assume that X is full rank. Where L denotes the loss on xj, the update for a single step of mini-batch SGD at time t with scalar learning rate η is:

w ) (2)

xj∈B

Mini-batch SGD iterates by randomly drawing mini-batches and updating model weights using the above equation until convergence is reached.

1.1.1 Minimum Norm Solution [2pt]

Recall Question 3.3 from Homework 1. For an overparameterized linear model, gradient descent starting from zero initialization finds the unique minimum norm solution w∗ such that Xw∗ = t. Let w0 = 0, d &gt; n. Assume mini-batch SGD also converges to a solution wˆ such that Xwˆ = t. Show that mini-batch SGD solution is identical to the minimum norm solution w∗ obtained by gradient descent, i.e., wˆ = w∗.

Hint: Is xj or B contained in span of X? Do the update steps of mini-batch SGD ever leave the span of X?

1.2 Adaptive Methods

We now consider the behavior of adaptive gradient descent methods. In particular, we will investigate the RMSProp method. Let wi denote the i-th parameter. A scalar learning rate η is used. At time t for parameter i, the update step for RMSProp is shown by:

) (3)

(4)

1.2.1 Minimum Norm Solution [1pt]

Consider the overparameterized linear model (d &gt; n) for the loss function defined in Section 1. Assume the RMSProp optimizer converges to a solution. Provide a proof or counterexample for whether RMSProp always obtains the minimum norm solution.

Hint: Compute a simple 2D case. Let x1 = [2,1], w0 = [0,0], t = [2].

1.2.2 [0pt]

Consider the result from the previous section. Does this result hold true for other adaptive methods (Adagrad, Adam) in general? Why might making learning rates independent per dimension be desirable?

2 Gradient-based Hyper-parameter Optimization

Often in practice, hyper-parameters are chosen by trial-and-error based on a model evaluation criterion. Instead, gradient-based hyper-parameter optimization computes gradient of the evaluation criterion w.r.t. the hyper-parameters and uses this gradient to directly optimize for the best set of hyper-parameters. For this problem, we will optimize for the learning rate of gradient descent in a regularized linear regression problem.

Specifically, given n pairs of input data with d features and scalar label (xi,ti) ∈ Rd ×R, we wish to find a linear model f(x) = wˆ⊤x with wˆ ∈ Rd and a L2 penalty, , that minimizes the squared error of prediction on the training samples. λ˜ is a hyperparameter that modulates the impact of the L2 regularization on the loss function. Using the concise notation for the data matrix X ∈Rn×d and the corresponding label vector t ∈Rn, the squared error loss can be written as:

.

Starting with an initial weight parameters w0, gradient descent (GD) updates w0 with a learning rate η for t number of iterations. Let’s denote the weights after t iterations of GD as wt, the loss as Lt, and its gradient as ∇wt. The goal is the find the optimal learning rate by following the gradient of Lt w.r.t. the learning rate η.

2.1 Computation Graph

2.1.1 [0.5pt]

Consider a case of 2 GD iterations. Draw the computation graph to obtain the final loss L˜2 in terms of w0,∇w0L˜0,L˜0,w1,L˜1,∇w1L˜1,w2,λ˜ and η.

2.1.2 [0.5pt]

Then, consider a case of t iterations of GD. What is the memory complexity for the forwardpropagation in terms of t? What is the memory complexity for using the standard back-propagation to compute the gradient w.r.t. the learning rate, ∇ηL˜t in terms of t? Hint: Express your answer in the form of O in terms of t.

2.1.3 [0pt]

Explain one potential problem for applying gradient-based hyper-parameter optimization in more realistic examples where models often take many iterations to converge.

2.2 Optimal Learning Rates

In this section, we will take a closer look at the gradient w.r.t. the learning rate. To simplify the computation for this section, consider an unregularized loss function of the form . Let’s start with the case with only one GD iteration, where GD updates the model weights from w0 to w1.

2.2.1 [1pt]

Write down the expression of w1 in terms of w0, η, t and X. Then use the expression to derive the loss L1 in terms of η.

Hint: If the expression gets too messy, introduce a constant vector a = Xw0−t

2.2.2 [0pt]

Determine if L1 is convex w.r.t. the learning rate η.

Hint: A function is convex if its second order derivative is positive

2.2.3 [1pt]

Write down the derivative of L1 w.r.t. η and use it to find the optimal learning rate η∗ that minimizes the loss after one GD iteration. Show your work.

2.3 Weight decay and L2 regularization

Although well studied in statistics, L2 regularization is usually replaced with explicit weight decay in modern neural network architectures:

wi+1 = (1 − λ)wi − η∇Li(X) (5)

In this question you will compare regularized regression of the form with unregularized loss, , accompanied by weight decay (equation 5).

2.3.1 [0.5pt]

Write down two expressions for w1 in terms of w0, η, t, λ, λ˜, and X. The first one using L˜, the second with L and weight decay.

2.3.2 [0.5pt]

How can you express λ˜ (corresponding to L2 loss) so that it is equivalent to λ (corresponding to weight decay)?

Hint: Think about how you can express λ˜ in terms of λ and another hyperparameter.

2.3.3 [0pt]

Adaptive gradient update methods like RMSprop (equation 4) modulate the learning rate for each weight individually. Can you describe how L2 regularization is different from weight decay when adaptive gradient methods are used? In practice it has been shown that for adaptive gradients methods weight decay is more successful than l2 regularization.

3 Convolutional Neural Networks

The last set of questions aims to build basic familiarity with convolutional neural networks (CNNs).

3.1 Convolutional Filters [0.5pt]

0

0 0

1 0

1 −1 −1 −1 ?

?

 ? ? ? ? ? ? ?

?

I = 1 1 1 1 0 J =  0 0 0 I ? ? ? ? ?

   

0 1 1 1 0 1 1 1 ? ? ? ? ?

0 0 1 0 0 ? ? ? ? ?

3.2 Size of Conv Nets [1pt]

CNNs provides several advantages over fully connected neural networks (FCNNs) when applied to image data. In particular, FCNNs do not scale well to high dimensional image data, which you will demonstrate below. Consider the following CNN architecture on the left:

The input image has dimension 32 × 32 and is grey-scale (one channel). For ease of computation, assume all convolutional layers only have 1 output channel, and use 3 × 3 kernels. Assume zero padding is used in convolutional layers such that the output dimension is equal to the input dimension. Each max pooling layer has a filter size of 2 × 2 and a stride of 2.

We consider an alternative architecture, shown on the right, which replaces convolutional layers with fully connected (FC) layers in an otherwise identical architecture. For both the CNN architecture and the FCNN architecture, compute the total number of neurons in the network, and the total number of trainable parameters. You should report four numbers in total. Finally, name one disadvantage of having more trainable parameters.

3.3 Receptive Fields [0.5pt]

In the previous architecture, suppose we replaced the 3 × 3 sized filter with a 5 × 5 sized filter in the convolution layers, while keeping max-pooling, stride, padding, etc. constant. What is the receptive field of a neuron after the second convolutional layer? List 2 other things that can affect the size of the receptive field of a neuron.
