Prompt

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