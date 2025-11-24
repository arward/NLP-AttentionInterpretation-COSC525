# NLP Attention Interpretation : COSC524

Brief : This project analyzes multi-head attention in BERT to understand which heads truly matter for cross-sentence reasoning. Using SNLI, we extract cross-sentence attention and measure variance across entailment, neutral, and contradiction labels to identify redundancy and explore pruning for efficiency.

<br />

Contributors :
Yousif Abdulhussein,
Margaret Kelley,
Alex Warden,
Jingtao Zhong 

<br />

Summary : Transformer models like BERT use multi-head self-attention to interpret relationships between tokens — but growing evidence shows that not all attention heads matter. Some heads play an essential semantic role, while others are redundant and contribute little to downstream task performance. This project explores which heads matter, where semantic reasoning emerges, and whether we can remove the useless ones without hurting accuracy.

Using BERT-base-uncased and the SNLI dataset, we:
* Extracted cross-sentence attention between premise + hypothesis pairs

* Computed a variance-based importance metric to evaluate how differently each head behaves across NLI labels (entailment, contradiction, neutral)

* Identified redundant heads, heavily concentrated in the first 3 layers

* Conducted head-pruning experiments to measure performance effects

Key Findings 
* ~22% of attention heads exhibit low semantic variance → prunable

* Pruning these heads causes <1% drop in accuracy and macro-F1

* Most semantic reasoning occurs in deeper layers

Overall, the work demonstrates that transformers are heavily over-parameterized, and selective pruning can improve efficiency without hurting predictive performance.

 <br />

Running the Notebook :
You can easily run this project using Google Colab:
1. Upload the notebook to Colab

2. Make sure runtime uses GPU: Runtime → Change runtime type → Hardware Accelerator: GPU

3. Install dependencies inside the first Colab cell: !pip install transformers datasets torch seaborn matplotlib

4. Run all cells.

5. For pruning experiments, adjust:
VARIANCE_THRESHOLD = 1e-6  (modify to prune more/less aggressively)
