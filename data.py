# data.py

problems = [
    {
        "id": 1,
        "symbolic": """DAG:
H → W → A
H ----→ A
Where:
H=1: husband sets alarm
H=0: husband doesn't set alarm
W=1: wife sets alarm
W=0: wife doesn't set alarm
A=1: alarm rings
A=0: alarm doesn't ring
Probability distribution:
P(A=1|H=0,W=0) = 0.08
P(A=1|H=0,W=1) = 0.54
P(A=1|H=1,W=0) = 0.41
P(A=1|H=1,W=1) = 0.86
P(W=1|H=0) = 0.74
P(W=1|H=1) = 0.24
Question: Is the direct effect H → A positive, disregarding the mediated effect through W?""",
        "verbal": "Imagine a self-contained, hypothetical world with only the following conditions, and without any unmentioned factors or causal relationships: Husband has a direct effect on wife and alarm clock. Wife has a direct effect on alarm clock. For husbands that don't set the alarm and wives that don't set the alarm, the probability of ringing alarm is 8%. For husbands that don't set the alarm and wives that set the alarm, the probability of ringing alarm is 54%. For husbands that set the alarm and wives that don't set the alarm, the probability of ringing alarm is 41%. For husbands that set the alarm and wives that set the alarm, the probability of ringing alarm is 86%. For husbands that don't set the alarm, the probability of alarm set by wife is 74%. For husbands that set the alarm, the probability of alarm set by wife is 24%. If we disregard the mediation effect through wife, would husband positively affect alarm clock?"
    },
   
]
