# LLaMA

Do training and prediction using LLaMA

## Running Steps 

<!-- 1. run `src/template_script/dataset_script.py` -->

- run `src/scripts/train_script_test.py`

- run `src/scripts/pred_script_test.py`

- run `src/analyse.py`

- run `src/eval_confidence_scores.py`


## Prompt

* GPT 生成潜台词

User:

    Argument 1:
    {arg1}

    Argument 2:
    {arg2}

    What's the implicit meaning between the arguments?
Bot:

    {subtext}
User:

    What is the discourse relation between Argument 1 and Argument 2?
    A. Comparison
    B. Contingency
    C. Expansion
    D. Temporal

    Answer:
Bot:

    {label and analysis}
User:

    Just output the answer directly.
Bot:

    {pure label}

* 蒸馏潜台词

Input:

    Argument 1:
    {arg1}

    Argument 2:
    {arg2}

    What's the implicit meaning between the arguments?
Output:

    {subtext}

* 筛选潜台词（辨别器）

Input:

    Argument 1:
    {arg1}

    Argument 2:
    {arg2}

    Implicit meaning:
    {subtext}

    Is this implicit meaning helpful to figure out the discourse relation between Argument 1 and Argument 2?

    Answer:
Output:

    {yes/no}

* 主任务

Input:

    Argument 1:
    {arg1}

    Argument 2:
    {arg2}

    {subtext}

    Question: What is the discourse relation between Argument 1 and Argument 2?
    A. Comparison
    B. Contingency
    C. Expansion
    D. Temporal

    Answer:
Output:

    {label}

## Options

* top-level

~~~
A. Comparison
B. Contingency
C. Expansion
D. Temporal
~~~

* pdtb2 second-level

~~~
A. Comparison.Concession
B. Comparison.Contrast
C. Contingency.Cause
D. Contingency.Pragmatic cause
E. Expansion.Alternative
F. Expansion.Conjunction
G. Expansion.Instantiation
H. Expansion.List
I. Expansion.Restatement
J. Temporal.Asynchronous
K. Temporal.Synchrony
~~~

* pdtb3 second-level

~~~
A. Comparison.Concession
B. Comparison.Contrast
C. Comparison.Similarity
D. Contingency.Cause
E. Contingency.Condition
F. Contingency.Purpose
G. Expansion.Conjunction
H. Expansion.Equivalence
I. Expansion.Instantiation
J. Expansion.Level-of-detail
K. Expansion.Manner
L. Expansion.Substitution
M. Temporal.Asynchronous
N. Temporal.Synchronous
~~~

* CoNLL16

~~~
A. Comparison.Concession
B. Comparison.Contrast
C. Contingency.Cause.Reason
D. Contingency.Cause.Result
E. Contingency.Condition
F. Expansion.Alternative
G. Expansion.Alternative.Chosen alternative
H. Expansion.Conjunction
I. Expansion.Exception
J. Expansion.Instantiation
K. Expansion.Restatement
L. Temporal.Asynchronous.Precedence
M. Temporal.Asynchronous.Succession
N. Temporal.Synchrony
~~~

