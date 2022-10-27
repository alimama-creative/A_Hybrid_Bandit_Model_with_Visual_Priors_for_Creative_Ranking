# Hybrid_Bandit_Model_with_Visual_Prior
A Hybrid Bandit Model with Visual Priors for Creative Ranking in Display Advertising

## Preparation for Training & Testing (for Mushroom)
1. Please clone this repository, and we call the directory that you cloned as ${HBM_ROOT}.

2. Install Tensorflow and requirements as below
    ```
    cd $(HBM_ROOT)
    pip install -r requirements.txt (python2)
    ```

3. To perform experiments, run the script by the following command
    ```
    python run.sh
    ```
    Logs are saved in `$(HBM_ROOT)/logs`.

4. Please find more details in our code.

Note that this implementation is a fork of [deep_contextual_bandits](https://github.com/tensorflow/models/tree/v2.3.0/research/deep_contextual_bandits) which is proposed by [DEEP_BAYESIAN_BANDITS_SHOWDOWN](https://openreview.net/pdf?id=SyYe6k-CW).
<!--4. Please download pretrained word vectors from [Glove](https://nlp.stanford.edu/projects/glove/).-->



