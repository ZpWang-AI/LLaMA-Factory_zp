ANS_WORD_LIST = {
    "pdtb2": [
        "although",    
        "nevertheless",
        "but",
        "however",     
        "because",     
        "so",
        "therefore",   
        "thus",
        "since",
        "instead",
        "or",
        "and",
        "furthermore",
        "instance",
        "first",
        "specifically",
        "previously",
        "then",
        "simultaneously"
    ],
    "pdtb3": [
        "although",
        "however",
        "but",
        "similarly",
        "because",
        "so",
        "if",
        "for",
        "and",
        "namely",
        "instance",
        "specifically",
        "by",
        "thereby",
        "instead",
        "previously",
        "then",
        "simultaneously"
    ],
    "conll": [
        "although",
        "but",
        "however",
        "as",
        "because",
        "consequently",
        "so",
        "thus",
        "instead",
        "rather",
        "and",
        "while",
        "example",
        "instance",
        "specifically",
        "then",
        "previously",
        "meanwhile"
    ]
}

ANS_LABEL_LIST = {
    "pdtb2": {
        "level1": [
            "Comparison",
            "Comparison",
            "Comparison",
            "Comparison",
            "Contingency",
            "Contingency",
            "Contingency",
            "Contingency",
            "Contingency",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Temporal",
            "Temporal",
            "Temporal"
        ],
        "level2": [
            "Comparison.Concession",
            "Comparison.Concession",
            "Comparison.Contrast",
            "Comparison.Contrast",
            "Contingency.Cause",
            "Contingency.Cause",
            "Contingency.Cause",
            "Contingency.Cause",
            "Contingency.Pragmatic cause",
            "Expansion.Alternative",
            "Expansion.Alternative",
            "Expansion.Conjunction",
            "Expansion.Conjunction",
            "Expansion.Instantiation",
            "Expansion.List",
            "Expansion.Restatement",
            "Temporal.Asynchronous",
            "Temporal.Asynchronous",
            "Temporal.Synchrony"
        ]
    },
    "pdtb3": {
        "level1": [
            "Comparison",
            "Comparison",
            "Comparison",
            "Comparison",
            "Contingency",
            "Contingency",
            "Contingency",
            "Contingency",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Temporal",
            "Temporal",
            "Temporal"
        ],
        "level2": [
            "Comparison.Concession",
            "Comparison.Concession",
            "Comparison.Contrast",
            "Comparison.Similarity",
            "Contingency.Cause",
            "Contingency.Cause",
            "Contingency.Condition",
            "Contingency.Purpose",
            "Expansion.Conjunction",
            "Expansion.Equivalence",
            "Expansion.Instantiation",
            "Expansion.Level-of-detail",
            "Expansion.Manner",
            "Expansion.Manner",
            "Expansion.Substitution",
            "Temporal.Asynchronous",
            "Temporal.Asynchronous",
            "Temporal.Synchronous"
        ]
    },
    "conll": {
        "level1": [
            "Comparison",
            "Comparison",
            "Comparison",
            "Contingency",
            "Contingency",
            "Contingency",
            "Contingency",
            "Contingency",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Expansion",
            "Temporal",
            "Temporal",
            "Temporal"
        ],
        "level2": [
            "Comparison.Concession",
            "Comparison.Contrast",
            "Comparison.Contrast",
            "Contingency.Cause.Reason",
            "Contingency.Cause.Reason",
            "Contingency.Cause.Result",
            "Contingency.Cause.Result",
            "Contingency.Cause.Result",
            "Expansion.Alternative.Chosen alternative",
            "Expansion.Alternative.Chosen alternative",
            "Expansion.Conjunction",
            "Expansion.Conjunction",
            "Expansion.Instantiation",
            "Expansion.Instantiation",
            "Expansion.Restatement",
            "Temporal.Asynchronous.Precedence",
            "Temporal.Asynchronous.Succession",
            "Temporal.Synchrony"
        ]
    }
}

SUBTYPE_LABEL2ANS_WORD = {
    "pdtb2": {
        "Comparison": "but",
        "Comparison.Concession": "although",
        "Comparison.Contrast": "but",
        "Contingency": "because",
        "Contingency.Cause.Reason": "because",
        "Contingency.Cause.Result": "so",
        "Contingency.Pragmatic cause": "since",
        "Expansion": "and",
        "Expansion.Alternative": "instead",
        "Expansion.Conjunction": "and",
        "Expansion.Instantiation": "instance",
        "Expansion.List": "first",
        "Expansion.Restatement": "specifically",
        "Temporal": "then",
        "Temporal.Asynchronous.Precedence": "then",
        "Temporal.Asynchronous.Succession": "previously",
        "Temporal.Synchrony": "simultaneously"
    },
    "pdtb3": {
        "Comparison": "but",
        "Comparison.Concession+SpeechAct.Arg2-as-denier+SpeechAct": "but",
        "Comparison.Concession.Arg1-as-denier": "although",
        "Comparison.Concession.Arg2-as-denier": "however",
        "Comparison.Contrast": "but",
        "Comparison.Similarity": "similarly",
        "Contingency": "because",
        "Contingency.Cause+Belief.Reason+Belief": "because",
        "Contingency.Cause+Belief.Result+Belief": "so",
        "Contingency.Cause+SpeechAct.Reason+SpeechAct": "because",
        "Contingency.Cause+SpeechAct.Result+SpeechAct": "so",
        "Contingency.Cause.Reason": "because",
        "Contingency.Cause.Result": "so",
        "Contingency.Condition+SpeechAct": "if",
        "Contingency.Condition.Arg1-as-cond": "if",
        "Contingency.Condition.Arg2-as-cond": "if",
        "Contingency.Purpose.Arg1-as-goal": "for",
        "Contingency.Purpose.Arg2-as-goal": "for",
        "Expansion": "and",
        "Expansion.Conjunction": "and",
        "Expansion.Disjunction": "and",
        "Expansion.Equivalence": "namely",
        "Expansion.Exception.Arg1-as-excpt": "and",
        "Expansion.Exception.Arg2-as-excpt": "and",
        "Expansion.Instantiation.Arg1-as-instance": "instance",
        "Expansion.Instantiation.Arg2-as-instance": "instance",
        "Expansion.Level-of-detail.Arg1-as-detail": "specifically",
        "Expansion.Level-of-detail.Arg2-as-detail": "specifically",
        "Expansion.Manner.Arg1-as-manner": "thereby",
        "Expansion.Manner.Arg2-as-manner": "by",
        "Expansion.Substitution.Arg2-as-subst": "instead",
        "Temporal": "then",
        "Temporal.Asynchronous.Precedence": "then",
        "Temporal.Asynchronous.Succession": "previously",
        "Temporal.Synchronous": "simultaneously"
    },
    "conll": {
        "Comparison": "but",
        "Comparison.Concession": "although",
        "Comparison.Contrast": "but",
        "Contingency": "because",
        "Contingency.Cause.Reason": "because",
        "Contingency.Cause.Result": "so",
        "Expansion": "and",
        "Expansion.Alternative.Chosen alternative": "instead",
        "Expansion.Conjunction": "and",
        "Expansion.Instantiation": "example",
        "Expansion.Restatement": "specifically",
        "Temporal": "then",
        "Temporal.Asynchronous.Precedence": "then",
        "Temporal.Asynchronous.Succession": "previously",
        "Temporal.Synchrony": "meanwhile"
    }
}