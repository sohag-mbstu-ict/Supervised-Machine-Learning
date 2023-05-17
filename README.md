# Simple Linear Regression

Simple linear regression is used to estimate the relationship between two quantitative variables.

### Simple linear regression example
You are a social researcher interested in the relationship between income and happiness. You survey 500 people whose incomes range from 15k to 75k and ask them to rank their happiness on a scale from 1 to 10.
Your independent variable (income) and dependent variable (happiness) are both quantitative, so you can do a regression analysis to see if there is a linear relationship between them.

![](Screenshot/simple_linear_regression/train.PNG)

In the above plot, we can see the real values observations in green dots and predicted values are covered by the red regression line. The regression line shows a correlation between the dependent and independent variable.

The good fit of the line can be observed by calculating the difference between actual values and predicted values. But as we can see in the above plot, most of the observations are close to the regression line, hence our model is good for the training set.

![](Screenshot/simple_linear_regression/test.PNG)

In the above plot, there are observations given by the blue color, and prediction is given by the red regression line. As we can see, most of the observations are close to the regression line, hence we can say our Simple Linear Regression is a good model and able to make good predictions.


# Multiple Linear Regression
Multiple Linear Regression is one of the important regression algorithms which models the linear relationship between a single dependent continuous variable and more than one independent variable.

In this piece, I am going to introduce the Multiple Linear Regression Model. Our problem is about modeling how R&D, administration, and marketing spendings and the state will influence the profit of a company. There are 50 startups data in our dataset.

```bash
x_opt=x[:,[0,1,2,3]]#copy the index 0 to 3 from x in x_opt
regressor_OLS=sm.OLS(endog=y,exog=x_opt.astype(float)).fit()
regressor_OLS.summary()
```

![](Screenshot/simple_linear_regression/MLP1.PNG)

Look at the highest p-values and remove it. In this condition x3(third  dummy variable has the highest one (0.767)

```bash
x_opt=x[:,[0,1,2]]#copy the index 0 to 2 from x in x_opt
regressor_OLS=sm.OLS(endog=y,exog=x_opt.astype(float)).fit()
regressor_OLS.summary()
```

![](Screenshot/simple_linear_regression/MLP2.PNG)

Look at the highest p-values and remove it. In this condition x1(first  dummy variable has the highest one (0.020)

```bash
x_opt=x[:,[0,2]]#copy the index 0 to 3 from x in x_opt
regressor_OLS=sm.OLS(endog=y,exog=x_opt.astype(float)).fit()
regressor_OLS.summary()
```

![](Screenshot/simple_linear_regression/MLP3.PNG)

Look at the highest p-values and remove it. In this condition x3(third  dummy variable has the highest one (0.767)

That’s it. The highest impact variable


# Polynomial Regression

Polynomial Regression is a regression algorithm that models the relationship between a dependent(y) and independent variable(x) as nth degree polynomial

## Need for Polynomial Regression:

If we apply a linear model on a linear dataset, then it provides us a good result as we have seen in Simple Linear Regression, but if we apply the same model without any modification on a non-linear dataset, then it will produce a drastic output. Due to which loss function will increase, the error rate will be high, and accuracy will be decreased.
So for such cases, where data points are arranged in a non-linear fashion, we need the Polynomial Regression model. We can understand it in a better way using the below comparison diagram of the linear dataset and non-linear dataset.

![](Screenshot/simple_linear_regression/polynomial.PNG)


# Decision Tree Classification Algorithm

Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.

It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.

### Attribute Selection Measures
If the dataset consists of N attributes then deciding which attribute to place at the root or at different levels of the tree as internal nodes is a complicated step. By just randomly selecting any node to be the root can’t solve the issue. If we follow a random approach, it may give us bad results with low accuracy.
For solving this attribute selection problem, researchers worked and devised some solutions. They suggested using some criteria like :

Entropy,

Information gain,

Gini index,

Gain Ratio,

Reduction in Variance

Chi-Square

### Entropy
Entropy is an information theory metric that measures the impurity or uncertainty in a group of observations. It determines how a decision tree chooses to split data.

### Information Gain
We can define information gain as a measure of how much information a feature provides about a class. Information gain helps to determine the order of attributes in the nodes of a decision tree.

The main node is referred to as the parent node, whereas sub-nodes are known as child nodes. We can use information gain to determine how good the splitting of nodes in a decision tree.

### Using Information Gain to Build Decision Trees

![](Screenshot/simple_linear_regression/DTC.PNG)




