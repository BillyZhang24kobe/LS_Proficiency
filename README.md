# ProLex ✍️: A Benchmark for Language Proficiency-oriented Lexical Substitution
<img src="assets/ProLex_figures.png" width="100%">

This is the repository for ProLex, a novel benchmark that evaluates system performances on language proficiency-oriented lexical substitution, a new task that proposes substitutes that are not only contextually suitable but also demonstrate advanced-level proficiency. For example:

> **Target word** `w`: `promotion (B2)`
>
> **Context** `s`: `This **promotion** has a beautiful and effective visual part, but they miss the real point: the product.`
>
> **Acceptable** w<sup>a</sup>: `advertising (A2), marketing (B1), publicity (B2), campaign (B1), advertisement (A2)`
>
> **Proficiency-oriented** w<sup>a</sup><sub>p</sub>: `publicity (B2)`

Note that the proficiency level of each word is indicated based on [Common
European Framework of Reference (CEFR)](https://en.wikipedia.org/wiki/Common_European_Framework_of_Reference_for_Languages). We refer to the CEFR Checker developed by [Cathoven AI](https://www.cathoven.com/en/cefr-checker/) to label the CEFR level of each word in ProLex.

In general, this repository offers:

1. The data format (CSV) for ProLex ✍️
2. An instruction tuning pipeline with task-specific synthetic data
3. A standardized evaluation pipeline

## News
- [2024/01] 🔥 We released the very first version of ProLex. Read the [paper](https://arxiv.org/abs/2401.11356) for more details!


## Table of Contents
- [Downloading the ProLex benchmark](#downloading-the-prolex-benchmark)
- [Environment settings](#environment-settings)
- [Instruction finetuning pipelines](#instruction-finetuning-pipelines)
- [Evaluating on ProLex](#evaluating-on-prolex)
- [Citation](#citation)
- [Questions](#questions)

## Downloading the ProLex benchmark
We prepare both the dev and test sets for ProLex ✍️. They can be downloaded from the following links:

- [ProLex development set](https://github.com/BillyZhang24kobe/LS_Proficiency/blob/main/data/dev/ProLex_v1.0_dev.csv)
- [ProLex test set](https://github.com/BillyZhang24kobe/LS_Proficiency/blob/main/data/test/ProLex_v1.0_test.csv)

### ProLex CSV format

ProLex is composed of quadruplets (w, s, w<sup>a</sup>, w<sup>a</sup><sub>p</sub>), each containing a target word, a context sentence, a list of acceptable substitutes, and a list of proficiency-oriented substitutes. We organize these contents into a CSV format. The columns are described as follows:

- `target word`: the target word as plain text.
- `Sentence`: the context sentence as plain text, with target word encompassed with asterisks.
- `acc_subs`: a list of acceptable substitutes annotated by human experts.
- `unacc_subs`: a list of unacceptable substitutes annotated by human experts.
- `prof_acc_subs`: a list of advanced proficiency-oriented substitutes from `acc_subs`.
- `prof_unacc_subs`: a list of low-proficiency substitutes removed from `acc_subs`.
- `t_words_cefr`: the CEFR level of the `target word`.
- `prof_acc_cefr`: the CEFR levels of the substitutes from `prof_acc_subs`.
- `prof_unacc_cefr`: the CEFR levels of the substitutes from `prof_unacc_subs`.

Note that we encoded the CEFR levels with integers ranging from 0 to 5. Please refer to the following mapping to derive the CEFR labels. Currently, there are some limitations of the CEFR checkers we are using. For example, it can not recognize certain words that are uncommon (e.g. gasoline). Also, it cannot recognize CEFR levels for phrases. However, to encourage vocabulary diversity of ProLex, we retain label `6` for accetapble but unknown words, and `None` for acceptable phrases, respectively.

```
0: A1
1: A2
2: B1
3: B2
4: C1
5: C2
6: unknown word
None: phrases
```

## Environment settings

## Instruction finetuning pipelines

## Evaluating on ProLex

## Citation

## Questions
