# NeuroSeqRet
Artifacts related to the AAAI 2022 paper "Learning Temporal Point Processes for Efficient Retrieval of Continuous Time Event Sequences".

## Requirements
Use a python-3.7 environment and install PyTorch v1.6.0, TorchVision v0.4.2, and Rank-Eval v0.0.1. More details are given in requirements.txt file. 

## Execution Instructions
### Dataset Format
We have also provided a dataset for your reference. For any new dataset, you need to structure it to different files containing the continuous-time event sequences (CTES) as follows:
```
test_query_ev test_query_ti train_corpus_ev train_corpus_ti train_query_ev train_query_ti
```
### File Details
Here, we provide the description of the files that are given in the dataset:
- train_query_ev.txt = The types of the events in the query CTES.
- train_query_ti.txt = The times of the events in the query CTES.

Similarly, we have the exact files for the corpus data and the test CTES.

### Running the Code
Use the following command to run NeuroSeqRet on the dataset. For example, to run the model on a dataset, use the command:
```
python main.py --dataset <dataset_name>
```
Once you run the command, the following procedure steps into action:
- NeuroSeqRet trains its retrieval system using the cross/self attention model.
- Later, it creates a dump for all CTES in the dataset. This dump is available in the 'Hash' folder with the name of the dataset followed by 'Embs'. This is a time-consuming process as these embeddings are calculated for all CTES in the corpus.
- Lastly, it will automatically initiate the evaluation process and will return results in therm of MPA and NDCG@10.

For calculating the hash-codes for all CTES data, you can run the following command
```
python hashing.py --dataset <dataset_name>
```
Once you run the above command, NeuroSeqRet loads the embedding dumps and creates the hash-codes and the collision files. In detail, it creates two files in 'Hash' folder:
- 'Codes' = The hash codes for all corpus CTES.
- 'Hashed' = The corpus sequences that were indexed for every test query, i.e., the collisions for each query. This is a list of the same size as the total number of test CTES.

## Citing
If you use this code in your research, please cite:
```
@inproceedings{aaai22,
 author = {Vinayak Gupta and Abir De and Srikanta Bedathur},
 booktitle = {Proc. of the 36th AAAI Conference on Artificial Intelligence (AAAI)},
 title = {Learning Temporal Point Processes for Efficient Retrieval of Continuous Time Event Sequences},
 year = {2022}
}
```

## Contact
In case of any issues, please send a mail to
```guptavinayak51 (at) gmail (dot) com```
