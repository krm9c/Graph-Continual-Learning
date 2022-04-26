#!/bin/bash
##!/bin/bash

# This is the model without any noise summary



# The theta implementations
# export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
# export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh" ]; then
#         . "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<
# conda activate torchRL



## The JLSE implementations
## mkdir one_peak_test
# python uq_v.py 
## --peak "one_peak"  --n_points 450000 --json_file "One_peak.json"  --output_model "one_peak_test/"  --input_flag 1  --output_flag 1 --factor_reset 10 --n_iterations  20
# source deactivate

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dh_posei
python multi_run_ICL.py
conda deactivate

