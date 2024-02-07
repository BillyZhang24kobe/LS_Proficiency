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

This command will save the target with context at `baselines/ParaLS_results/<split>/processed.tsv`, and save gold substitute files for both acceptable substitutes (`baselines/ParaLS_results/<split>/gold_acc.tsv`) and proficiency-oriented substitutes (`baselines/ParaLS_results/<split>/gold_prof_acc.tsv`). We can copy these three files to the ParaLS repo and put it under `data/LSPro/test`. In our [fork](https://github.com/cynic01/ParaLS/tree/main/data/LSPro/test) it is already there.

## Run ParaLS
We have created a fork of the ParaLS repo [here](https://github.com/cynic01/ParaLS) mainly to store our replication results and modify their code for ease of use. Please clone this repository and run it according to the README instructions. Alternatively, you can use the code from the original authors [here](https://github.com/qiang2100/ParaLS).

The results file in the ParaLS repo will be under a path like `lspro_search_results/log.###/lspro.out.embed.0.02.oot` if our fork is used. Copy this file to the directory `baselines/ParaLS_results/test` in this repo before proceeding to the next step.

## Transform ParaLS output into ProLex format
We can use `transform_dataset.py` to transform the ParaLS output back to the ProLex format.

```
python3 transform_dataset.py --direction parals_to_prolex --split test
```

The converted outputs are stored under `outputs/ParaLS_test.csv`.

## Evaluating on ProLex
We can now evaluate the ParaLS model outputs using `evaluate.py`.

```
python3 evaluate.py --model_name_or_path outputs/ParaLS_test.csv --data_path data/test/ProLex_v1.0_test.csv
```

## Questions
Please reach out to us at billyzhang@cs.columbia.edu if you have any questions in using our benchmark. If you find an issue in either the source code or dataset, please feel free to create a pull request and make contribution to the benchmark!