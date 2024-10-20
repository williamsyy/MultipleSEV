# Improving Decision Sparsity

This repository is the official implementation of [Improving Decision Sparsity](https://arxiv.org/abs/2030.12345), which is accepted by NeurIPS 2024. This paper provides a Sparse Explanation Values (SEV) variants for the [original SEV Paper](https://github.com/williamsyy/SparseExplanationValues) by introducing multiple reference for the SEV calculation.

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Sparse Explanation Values

The usage of SEV Variants can reference the Experiment Part of SEV. The general usage of ClusterSEV can be used as follows:

```python
from SEV.ClusterSEV import ClusterSEV
from SEV.data_loader import data_loader
from SEV.Encoder import DataEncoder
from sklearn.linear_model import LogisticRegression

# get the dataset and the negative population
X,y,X_neg = data_loader("adult")

# set up the parameters for the clustering
num_clusters = 5
m = 1.01

# preprocessing the dataset
encoder = DataEncoder(standard=True)
# fit the encoder with the negative population
encoder.fit(X_neg)
# transform the whole dataset
encoded_X = encoder.transform(X)

# construction the model
lr = LogisiticRegression()
lr.fit(encoded_X,y)

sev = ClusterSEV(model,encoder, encoded_X.columns, encoded_X_neg, n_clusters=num_clusters,m=m)

# for explaining the whole dataset, "plus" for SEV+, "minus" for SEV-
for i in tqdm(range(encoded_X)):
    Xi = encoded_X.iloc[i].values.reshape(1,-1)
    if model.predict(Xi) != 1:
        cluster_sev.append(0)
        continue
    sev_num,diff = sev.sev_cal(Xi,X_test_emb[i].reshape(1,-1),mode="minus")

# get the number of sev for this instance
sev_num = sev.sev_cal(np.array(encoded_X.iloc[0]).reshape(1,-1),mode="plus")
print("The SEV Value for instance 0 is %d."%sev_num)
# get the features can be used in this explanation
features = sev.sev_count(np.array(encoded_X.iloc[0]).reshape(1,-1),mode="plus",choice=sev_num)
print("The feature used in this explanation are %s."%features)
```

## Experiments

All the experiments within the paper are shown in the Experiment Folder, to run a Experiment for Cluster-based SEV, run

```bash
python Experiment\ 1\ Cluster.py --data adult --method l1lr --iterations 10
```

## Citation

To use our method please cite both the original SEV paper and the SEV Variants Paper:

```
@article{sun2024sparse,
  title={Sparse and Faithful Explanations Without Sparse Models},
  author={Sun, Yiyang and Chen, Zhi and Orlandi, Vittorio and Wang, Tong and Rudin, Cynthia},
  journal={Society for Artificial Intelligence and Statistics (AISTATS)},
  year={2024}
}

@article{sun2024improving,
  title={Improving Decision Sparsity},
  author={Sun, Yiyang and Wang, Tong and Rudin, Cynthia},
  journal={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

