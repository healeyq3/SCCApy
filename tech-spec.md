---
Created: 4/29/24
Updated: 4/29/24
---

# Overarching Principles/Goals

1. Provide rich experimentation.
    
    a. Experimenters can easily run trials for varying $s_i, n_i$, $i = 1, 2$ from the command line (i.e. experimenters don't have to edit any Python code to run one or more experiments).

    b. All experiments are saved

    c. Advanced experimenters can replace the default bounding algorithms with their own without having to adjust the codebase (they will have to write their own experimentation file, however.)

2. Implement the branch and bound framework to facilitate (1), but also 

3. (make the framework easy to use for global optimization)

    a. need to make it easy to use local algorithms.

# TODOS

Architecture Questions
1. Move the bounding specifications to the solve function (allow to )

Need to ensure whatever object is returned by the solve function can be used to generate statistics

1. Write tests for 

# Architecture

Note: Variable Fixing


Abstracting the bounding methods (and objective function)
- ProblemData is used to keep track of a Problem instance's shape data, as well as any additional data that particular bounding algorithms might need (in our case, the matrices $A$, $B$, and $C$.)

# Experimentation Features (Implementation Todos)

Would it be more user friendly to run from a python file or the command line...probably a python file

provide a list

# Basics

Statistics being kept track of
- GAP (after LB was run)
- number iterations (kept track of by tree)
- number of nodes (kept track of by Node Class property)
- 

Statistics to keep track of
- initial GAP
- total solve time (including lower bound finding and varfix)
- num solved == num nodes
- num iterations == k
- 

Make experimentation easy
- declare a problem
- 

ProblemData -> main

Main functionality
- check

Define custom exceptions

