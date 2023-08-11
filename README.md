# Ada2Fair

**Ada2Fair** aims to promote two-sided fairness with adaptive weights for providers and customers in recommendation.

This repository is the official implementation of our paper under review (WSDM'24).

## Ada2Fair: Two-sided fairness-aware training framework

![](assets/model.pdf)

Ada2Fair involves two stages: how to generate weights in stage I and and how to utilize weights in stage II. The former generates weights as a prerequisite for the latter, while the latter utilizes weights as our core objective.

To learn two-sided fairness-aware weights during model training in a general way, we propose an adaptive weight generator to produce fairness-aware weights, and devise the strategies for two-sided fairness-aware weighting based on dynamic recommendation results. Then, a fairness-aware weight adapter is futher utilized to guide the learning of the weight generator.

## Quick Start

1. Unzip dataset files.
    ```bash
    cd ada2fair/dataset/Book-Crossing/; unzip Book-Crossing.zip
    cd ada2fair/dataset/Amazon_Video_Games/; unzip Amazon_Video_Games.zip
    cd ada2fair/dataset/BeerAdvocate/; unzip BeerAdvocate.zip
    ```
2. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
3. Evaluate the performance of BPRMF model with and without our approach on Book-Crossing dataset.
    ```bash
    # without Ada2Fair (the typical accuracy-focused method)
    cd ada2fair/
    python main.py --model=BPR --dataset=Book-Crossing
    ```
    
    ```bash
    # with Ada2Fair (the two-sided fairness-aware method)
    cd ada2fair/
    python main.py --model=BPR --dataset=Book-Crossing --fairness_type=ada2fair
    ```

## Hyper-parameter tuning

