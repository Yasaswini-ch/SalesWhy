# Evaluation

## SalesWhy (TabPFN-local + deterministic cause attribution)

| cause          |   times_injected |   correctly_named |   accuracy |
|:---------------|-----------------:|------------------:|-----------:|
| freight_spike  |               10 |                 2 |        0.2 |
| price_increase |               10 |                 0 |        0   |
| stockout       |               10 |                 9 |        0.9 |


Abstain rate: 3.3% | Miss rate (not flagged at all): 20.0% | Overall accuracy on flagged+named windows: 36.7%


## Baseline (a): moving-average predictor + same cause attribution

| cause          |   times_injected |   correctly_named |   accuracy |
|:---------------|-----------------:|------------------:|-----------:|
| freight_spike  |               10 |                 0 |        0   |
| price_increase |               10 |                 0 |        0   |
| stockout       |               10 |                 9 |        0.9 |


Abstain rate: 6.7% | Miss rate: 23.3% | Overall accuracy: 30.0%


## Baseline (b): generic LLM asked "why did sales drop?"

Same raw CSV context as the deterministic pipeline sees (dates, revenue, price, freight, active products, orders), one plain question, no structured output, no retries beyond transient API errors. The model's free-text answer is classified into a cause by keyword matching after the fact -- the same thing a human reading the answer would do.

| true_cause     |   times_injected |   correctly_named |   accuracy |
|:---------------|-----------------:|------------------:|-----------:|
| freight_spike  |               10 |                 4 |        0.4 |
| price_increase |               10 |                 3 |        0.3 |
| stockout       |               10 |                 9 |        0.9 |


Abstain rate: 23.3% | Overall accuracy: 53.3% | Mean latency: 5.47s | Tokens used: 20866 prompt + 967 completion across 30 calls


**This is the single most surprising result in this evaluation: the generic LLM (53.3% accuracy) outperformed SalesWhy's own deterministic pipeline (36.7%).** On inspection, the LLM was especially strong at stockout (90%, matching SalesWhy) because it reasons in natural language about 'zero active products' however it's phrased, while our keyword-based scoring of *its* answers initially missed that phrasing entirely -- a bug in how we graded it, not in what it said (fixed; see DEVLOG). On price/freight causes, where SalesWhy struggles on sparse data (0-20% accuracy), the generic LLM did moderately better (30-40%), but it also has a different failure mode worth flagging: it sometimes correctly spots that revenue fell but misattributes it to the proximate symptom (order count dropping to zero) rather than the underlying price/freight change, which is a confident-sounding wrong answer, not an abstain. SalesWhy's deterministic attribution never does this by construction -- it can only ever point at price, freight or product-count, never 'demand fell for some other reason' -- which trades some accuracy for a narrower, more honest failure mode.


## Note on abstain composition

Every injected drop in this eval has a known, explainable cause by construction, so an abstain here is always a miss, never a 'truly unexplainable' drop. The not-sure rate on real, non-synthetic seller data (where some drops genuinely have no cause in the CSV) still needs separate reporting once real seller data is available.
