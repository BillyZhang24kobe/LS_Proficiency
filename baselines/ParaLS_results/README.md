# Evaluating ParaLS models using ProLex

This folder contains the code necessary to reproduce the ParaLS model results as evaluated using ProLex.

## Steps
- [Transform ProLex test set into ParaLS format](#Transform-ProLex-test-set-into-ParaLS-format)
- [Run ParaLS](#Run-ParaLS)
- [Transform ParaLS output into ProLex format](#Transform-ParaLS-output-into-ProLex-format)
- [Evaluating on ProLex](#evaluating-on-prolex)
- [Questions](#questions)

## Transform ProLex test set into ParaLS format
We have prepared a script `transform_dataset.py` to help with transforming between different dataset formats. Run the following command to convert our test set from the ProLex format to the ParaLS format.

```
python3 transform_dataset.py --direction prolex_to_parals --split test
```

## Run ParaLS
We have created a fork of the ParaLS repo [here](https://github.com/cynic01/ParaLS) mainly to store our replication results and modify their code for ease of use. Please clone this repository and run it according to the README instructions. Alternatively, you can use the code from the original authors [here](https://github.com/qiang2100/ParaLS).

## Transform ParaLS output into ProLex format
We can use `transform_dataset.py` to transform the ParaLS output back to the ProLex format.

```
python3 transform_dataset.py --direction parals_to_prolex --split test
```

## Evaluating on ProLex
We can now evaluate the ParaLS model outputs using `evaluate.py`.

```
python3 evaluate.py --model_name_or_path outputs/ParaLS_test.csv --data_path data/test/ProLex_v1.0_test.csv
```

## Questions
Please reach out to us at billyzhang@cs.columbia.edu if you have any questions in using our benchmark. If you find an issue in either the source code or dataset, please feel free to create a pull request and make contribution to the benchmark!