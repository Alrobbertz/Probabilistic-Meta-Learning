# Probabilistic-Meta-Learning

## Data

* [Numenta Anomaly Benchmark](https://github.com/numenta/NAB)
* [Skoltech Anomaly Benchmark](https://github.com/waico/SKAB)
## Code

Code follows paper [Meta-Learning Probabilistic Inference For Prediction](https://arxiv.org/abs/1805.09921) and uses implmentation from GitHub [Versa](https://github.com/Gordonjo/versa). 

## To Run

Samples fom Bash to execute the run_classifier.py file.

### Omniglot

`
python run_classifier.py -d Omniglot --iterations 200
`

### NAB

`
python run_classifier.py -d NAB --d_theta 128 --shot 5 --way 2 --test_way 2 --iterations 10000
`

### SKAB

`
python run_classifier.py -d SKAB --d_theta 128 --shot 5 --way 2 --test_way 2 --iterations 10000
`