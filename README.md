# neural-sum

Simple neural network written in Python that learns to add two numbers.

## How it works

The program trains a neural network with:

* 2 inputs
* 1000 hidden neurons
* 1 output

The network learns to approximate the sum of two numbers using gradient descent.

## Requirements

* Python 3

You can check your version in the terminal (macbook) with:

python3 --version

## Usage

Run the program:

python3 neural_sum.py

Then enter two numbers and the network will train to predict their sum.

## Example

Input:
627
927

Output (approx):
1554

## Adjustable parameters

Inside the code you can modify several parameters:

* `N_HIDDEN` → number of hidden neurons in the network
* `LEARNING_RATE` → learning speed of the network
* `ITERATIONS` → number of training iterations
* `ESCALA` → normalization factor for inputs

Example in the code:

```python
N_HIDDEN = 1000
LEARNING_RATE = 0.000126
ITERATIONS = 20000
ESCALA = 100.0
```

## Learning rate rule

If you change the number of hidden neurons (`N_HIDDEN`), you should also adjust the learning rate.

Use the following formula:

```
LEARNING_RATE = 0.001 / (N_HIDDEN^0.3)
```

This keeps the training stable when increasing the network size.

## Learning rate example

Example using the formula:

```
LEARNING_RATE = 0.001 / (N_HIDDEN^0.3)
```

If we choose:

```
N_HIDDEN = 1000
```

Then:

```
1000^0.3 ≈ 7.94
```

So:

```
LEARNING_RATE = 0.001 / 7.94
LEARNING_RATE ≈ 0.000126
```

This is the value used in the example code.


## Iterations rule

If you increase the number of hidden neurons (`N_HIDDEN`), you should also increase the number of training iterations (`ITERATIONS`).

Larger networks usually need more iterations to converge properly.

## Author

YJNWI
