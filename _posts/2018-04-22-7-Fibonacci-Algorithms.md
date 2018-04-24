---
layout: post
title:  "7 Fibonacci Algorithms"
date:   2018-04-20
categories: Math Algorithms Python
---

The Fibonacci sequence is an important integer sequence defined by the following recurrence relation:

$$F(n) =
\begin{cases}
0, & \text{if $n = 0$}\\
1, & \text{if $n = 1$}\\
F(n - 1) + F(n - 2), & \text{if $n > 1$}
\end{cases}$$

The Fibonacci sequence is often used in introductory computer science courses to explain recurrence relations, dynamic programming, and proofs by induction. Because the Fibonacci sequence is central to several core computer science concepts, the following programming problem has become fairly popular in software engineering interviews:

### Given an input $$n$$, return the $$n$$th number in the Fibonacci sequence


Below I've listed 7 solutions to this problem, ranked from least efficient to most. These different solutions illustrate different programming and algorithmic techniques that can be used to solve other problems. You'll notice that many of these solutions build from previous solutions.

Note: I have recorded the space and time complexities for each algorithm below. In these equations, I refer to n as the *value* of n, not its size, which is sometimes preferred when discussing algorithms with numerical inputs. Also, I will treat all integer operations (such as addition, multiplication, comparison, etc.) as constant time operations, and I will regard the space being used by all integers as constant. For large integers, these assuptions may not suffice. Finally, input validation was not included in any of these code snippets.


# Solution 1 - Recursion

Runtime complexity: $$O(\phi^n)$$, where $$\phi$$ is the golden ratio ($$\phi \simeq 1.618...$$)

Space complexity: $$O(n)$$

```python

def fib1(n):
    """
    input: int
    output: int

    Returns the n'th Fibonacci number using recursion
    """

    if n == 0 or n == 1:
        return n

    return fib1(n - 1) + fib1(n - 2)

```

This algorithm is nearly a literal translation of the Fibonacci recurrence relation shown in the beginning. `fib1` will use $$O(n)$$ space on the call stack due to its recursive implementation - the longest chain of unresolved calls occurs when `fib1(n)` calls `fib1(n - 1)`, which calls `fib1(n - 2)` ... until we reach the base case `fib1(1)`. The length of this chain is n, so the space utilization scales as $$O(n)$$

To find the runtime complexity of this algorithm, one can first note that each call to `fib1` does constant work by itself (ignoring the work being done by the recursive calls) - it checks that the input is 0 or 1, then either immediately returns a value, or makes two more calls before returning a value.

So, if we count the number of calls that this algorithm makes for any n, we will find its runtime complexity.

Let $$T(n)$$ refer to the number of calls to `fib1` that are required to evaluate `fib1(n)`

When `fib1(0)` or `fib1(1)` is called, `fib1` doesn't make any additional recursive calls, so $$T(0)$$ and $$T(1)$$ must equal 1.

For any other n, when `fib1(n)` is called, `fib1(n - 1)` and `fib1(n - 2)` are also called. So, $$T(n)$$ must equal $$1 + T(n - 1) + T(n - 2)$$.

Using more advanced mathematical techniques, like generating functions, one can solve the recurrence $$T$$, and find that $$T$$ scales as $$O(\phi)$$

# Solution 2 - Top Down Dynamic Programing

Runtime complexity: $$O(n)$$

Space complexity: $$O(n)$$

```python
def fib2(n, memo = {}):
    """
    intput: int
    output: int

    Returns the n'th Fibonacci number using top-down dynamic programming
    """

    if n == 0 or n == 1:
        return n

    if n not in memo:
        memo[n] = fib2(n - 1, memo) + fib2(n - 2, memo)

    return memo[n]
```

This function looks very similar to `fib1`, but its complexity appears to have dramatically reduced. This is due to the magic of dynamic progamming.

`memo` is passed through each of the calls to `fib2`, recording the function's output into a dictionary before it returns. When this is done, if `fib2(i, memo)` has been computed once, subsequent calls to `fib2(i, memo)` take only constant time to be pulled from the dictionary.

For any n greater than 1, when `fib2(n)` is called, after `fib2(n - 1, memo)` returns, `fib2(n - 2, memo)` is called. However through the course of computing `fib2(n - 1, memo)`, the key n - 2 must have been added to `memo`. So, the second recursive call made in `fib2(n, memo)` will always be a constant time lookup. Since only one recursive call has a chance of making subsequent recursive calls, and since n is being decremented by a fixed amount (which is 1) in these calls, `fib2` must run in linear time.

Through the course of its execution, `memo` will eventually contain all keys in the range `2,3, ... n`. So, `memo` uses $$O(n)$$ space. $$O(n)$$ space is used on the call stack as well due to recursion as well.

# Solution 3 - Bottom Up Dynamic Programming

Runtime complexity:  $$O(n)$$

Space complexity: $$O(n)$$

```python
def fib3(n):
    """
    input: int
    output: int

    Returns the n'th Fibonacci number using bottom up dynamic programming
    """

    memo = [0, 1]
    
    while len(memo) <= n:
        memo.append(memo[-1] + memo[-2])

    return memo[n]
```


This algorithm is similar to the previous one, but it is implemented iteratively instead of recursively.

When `fib3(n)` is initially called, `memo` contains a list of the 0th and 1st Fibonacci numbers in order (i.e. `[0,1]`)

In each iteration of the while loop, a new Fibonacci number is computed by adding together the previous two Fibonacci numbers stored in `memo`. This new Fibonacci number is appended to the end of `memo`, so that `memo` now contains one more Fibonacci number, which corresponds to the last index of the list. When `memo` contains an nth element, it will be the nth Fibonacci number, and it will be returned by the method.

Since n + 1 elements are sequentially added to `memo` (including the 0th Fibonacci number), this function uses $$O(n)$$ time and space. (Python's lists are backed by arrays, guaranteeing an amortized $$O(1)$$ append operation, and $$O(1)$$ accesses for all indexes). This function is not recursive, so it only uses constant space on the call stack.

# Solution 4 - Tail Recursion

Runtime complexity:  $$O(n)$$

Space complexity: $$O(n)$$ (or $$O(1)$$ if your language implements tail call optimization)

```python
def fib4(n, i = 0, current = 0, next = 1):
    """
    input: int
    output: int

    Returns the n'th Fibonacci number using tail recursion
    """
    if n == i:
        return current

    return fib4(n, i + 1, next, current + next)
```

Python's standard implementation (CPython) doesn't implement tail call optimization, but I figured I would include this solution anyways.

In this solution, we pass in the two initial Fibonacci numbers by default. In the recursive calls that follow, i acts as a counter that counts up to n. As an invariant, `current` will contain the ith Fibonacci number, and `next` will contain the i + 1th Fibonacci number. If i equals n, the function returns `current`. Otherwise, a tail call is made, where i is incremented, and `next` is passed in to `current` while the following Fibonacci number is calculated and passed in to `next` to maintain the invariant.

Through these recursive calls, i is incremented from 0 to n, and each call does constant work, so this function runs in $$O(n)$$ time. Due to the recursive calls, this function should use $$O(n)$$ space on the call stack. However, if we were to rewrite this algorithm in a language that does typically implement tail call optimization (e.g. Haskell, Scheme, Scala, etc.), the recursive calls would not take up extra space on the call stack, making the function use only $$O(1)$$ space.

# Solution 5 - Iterative Solution

Runtime complexity:  $$O(n)$$

Space complexity: $$O(1)$$

```python
def fib5(n):
    """
    input: int
    output: int

    Returns the n'th Fibonacci number iteratively without extra space
    """
    current = 0
    next = 1

    for i in range(n):
        new = next + current
        current = next
        next = new

    return current
```


This solution is one of the simplest and most efficient algorithms on this list. It is similar to Solution 4, in that it keeps track of just two Fibonacci numbers, and uses them to compute the next Fibonacci number without holding on to all of the prior Fibonacci numbers in a list or a map.

This function has a loop that runs through n iterations before returning, and each iteration does constant work, making the algorithm run in $$O(n)$$ time. This function doesn't use any extra data structures, and it isn't recursive, so we can say that it uses $$O(1)$$ space.

# Solution 6 - Using Recurrence Matrix

Runtime complexity:  $$O(log(n))$$

Space complexity: $$O(1)$$

```python
def matrix_multiply(A, B):
    """
    inputs: List[List[int]] A & B (inner dimensions of A and B should match)
    output: List[List[int]]

    Returns the matrix product A * B
    """
    a_rows = len(A)
    a_cols = len(A[0])

    b_rows = len(B)
    b_cols = len(B[0])

    output = [ [0] * b_cols for i in range(a_rows) ]

    for i in range(a_rows):
        for j in range(b_cols):
            for k in range(b_rows):
                output[i][j] += A[i][k] * B[k][j]
    return output

def identity_matrix(n):
    """
    input: int
    output: List[List[int]]

    Returns n x n identity matrix
    """

    ident = [[0] * n for i in range(n)]
    for i in range(n):
        ident[i][i] = 1
    return ident

def fast_matrix_power(A, n):
    """
    inputs: List[List[int]] A (should be square), int n
    output: List[List[int]]

    Returns the matrix power A ** n
    """

    m = n
    B = [row[:] for row in A]
    output = identity_matrix(len(A))
    output

    #loop invariant: output * (B ** m) == A ** n
    while m != 0:
        if m % 2 == 1:
            output = matrix_multiply(output, B)
            m -= 1
        else:
            B = matrix_multiply(B, B)
            m /= 2

    return output

def fib6(n):
    """
    input: int
    output: int

    Returns the n'th Fibonacci number using a matrix multiplication algorithm
    """
    fib_matrix = [[0, 1], [1, 1]]
    fib_matrix_n = fast_matrix_power(fib_matrix, n)

    base_vector = [[0], [1]]
    product = matrix_multiply(fib_matrix_n, base_vector)

    return product[0][0]

```

This is the most mathematical solution on the list, and it requires some basic understanding of matrix multiplication.

Consider the following matrix $$F$$:

$$F = 
\begin{bmatrix}
  0 & 1 \\
  1 & 1
\end{bmatrix}
$$

If $$f_{i-1}$$ and $$f_{i-2}$$ are the i-1th and the i-2th Fibonacci numbers, the following matrix product can give us the ith Fibonacci number ($$f_{i}$$):

$$
F * 
\begin{bmatrix}
    f_{i - 2} \\
    f_{i - 1}
\end{bmatrix}

=

\begin{bmatrix}
    f_{i - 1} \\
    f_{i - 1} + f_{i - 2}
\end{bmatrix}

=

\begin{bmatrix}
    f_{i - 1} \\
    f_{i}
\end{bmatrix}
$$

Using this property of $$F$$, one can find $$f_{n}$$ by computing the product

$$
F^n *
\begin{bmatrix}
    f_0 \\
    f_1
\end{bmatrix}

= 
F^n * 
\begin{bmatrix}
    0 \\
    1
\end{bmatrix}
$$

and returning the first element of the vector.

The key here is to compute F^n using the successive square method, as done in the function `fast_matrix_power`. Using this algorithm, $$F^n$$ is computed in $$O(log(n))$$ time (Note that for a fixed matrix size, the matrix muliplication algorithm takes a constant amount of time). Then, $$F^n$$ is multiplied by $$\begin{bmatrix} 0 \\ 1 \end{bmatrix}$$, and the top element of the product is returned.

This algorithm uses $$O(log(n))$$ to compute $$F^n$$, and constant space, as the intermediary matricies in `fast_matrix_power` can be reclaimed by the garbage collecter.  

# Solution 7 - Lookup Table - 32 bit signed integers only!

Runtime complexity:  $$O(1)$$

Space complexity: $$O(1)$$

```python
fib_list = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584,4181,
            6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229, 832040,
            1346269, 2178309, 3524578, 5702887, 9227465, 14930352, 24157817, 39088169, 63245986,
            102334155, 165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903]

def fib7(n):
    """
    input: int n, where n < 47
    output: int

    Returns the n'th Fibonacci number using a list lookup
    """
    return fib_list[n]
```

This function will work strictly in the case that we're dealing with 32 bit signed integers (which could be a constraint in languages like Java, C/C++, etc.)

The Fibonacci sequence grows very quickly. So fast, that only the first 47 Fibonacci numbers fit within the range of a 32 bit signed integer. This method requires only a quick list lookup to find the nth Fibonacci number, so it runs in constant time. Since the list is of fixed length, this method runs in constant space as well.
