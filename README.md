# HACKODE

## DATA
[References](data/references.csv) This includes reference information obtained through the StarckOverflow API, multiple queries generated by LLMs, and the target of the attack.

[Prompt Templates](data/prompt_templates.txt) The prompt templates for generating attack sequences.

[Instructions](data/instructions.txt) The instructions for generating attack sequences.

## Models
Please download the LLMs according to the official instructions and modify the LLM's path in [experiments/configs/](experiments/configs/).
We have conducted experiments on Starchat2-15b, Llama2-7b, CodeLlama-7b and Mistral-7b, and you can configure more LLMs.
Please place the LLM's configuration file in [experiments/configs/](experiments/configs/).

## Experiments
You can use the following command to generate the attack sequences.
```
cd experiments/launch_scripts
./gen_attack_seq.sh mistral references
