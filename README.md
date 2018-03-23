# GraphSAGE Implementation for Characterizing and Detecting Hateful Users on Twitter


#### Requirements

pytorch >0.2 is required.

#### Running examples

1. Download the dataset from [kaggle](https://www.kaggle.com/manoelribeiro/hateful-users-on-twitter).

2. You need to create folders `hate` and `suspended` as follows:

       |- hate
       |---- users.edges
       |---- users_all_neighborhood.csv
       |---- users_hate_all.content
       |---- users_hate_glove.content
       |- suspended
       |---- users.edges
       |---- users_all_neighborhood.csv
       |---- users_suspended_all.content
       |---- users_suspended_glove.content
       
3.  Execute `python -m graphsage.model` to replicate everything.
