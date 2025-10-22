#!/bin/bash
#SBATCH -JToolEvaluation                         # Job name
#SBATCH --account=gts-tbaluta3                  # charge account
#SBATCH -N1                                     # Number of nodes and cores per node required 
#SBATCH --gres=gpu:L40S:2                       # GPU type (V100) and number of GPUs 
#SBATCH --mem-per-gpu=48GB                      # Memory per CPU core, 8 CPUs/GPU
#SBATCH --time=02:00:00                         # Time limit hrs:min:sec
#SBATCH -qembers                                # QOS Name
#SBATCH -oReport-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=ychauhan9@gatech.edu        # E-mail address for notifications

export HUGGINGFACE_TOKEN=$(cat /storage/scratch1/6/ychauhan9/ToolCommander/hgtoken.env)

cd ~/scratch/ToolCommander

module load cuda
module load anaconda3

conda activate toolcommander

export CUDA_VISIBLE_DEVICES=0,1

# Define the paths to the assets, corpus, and tool root directory here
ASSETS_PATH_BASE="./data"
CORPUS_PATH="./data/toolbench/corpus.tsv"
TOOL_ROOT_DIR="./data/toolbench/tools"
RESULTS_DIR="./attack_results"

# Define the retrievers to attack here
retrievers=("facebook/contriever-msmarco" "ToolBench/ToolBench_IR_bert_based_uncased")

# Define the assets to attack with here
assets=("g1_train_a.json" "g1_train_b.json" "g1_train_c.json")

#mkdir -p $RESULTS_DIR

#for asset in "${assets[@]}"; do
#    asset_name="${asset%.json}"
#    asset_results_dir="${RESULTS_DIR}/${asset_name}"
#    mkdir -p "$asset_results_dir"

#    for retriever in "${retrievers[@]}"; do
#        echo "Attacking with target retriever: $retriever using asset: $asset..."
#        retriever_name=$(echo $retriever | sed 's/\//_/g')
#        result_file="${asset_results_dir}/attack_results_${retriever_name}.json"

        # Execute the attack
#        srun python attack.py attack \
#            --model_retriever="$retriever" \
#            --assets_path="${ASSETS_PATH_BASE}/${asset}" \
#            --corpus_tsv_path="$CORPUS_PATH" \
#            --tool_root_dir="$TOOL_ROOT_DIR" \
#            --attack_result_save_path="$result_file" \
#            --n_samples=256 \
#            --topk=32 \
#            --epochs=64 \
#            --tag_length=64
#    done
#done


#echo "All attacks completed!"

export CUDA_VISIBLE_DEVICES=0,1

export HF_HOME="/storage/scratch1/6/ychauhan9/huggingface"
export TRANSFORMER_CACHE="$HF_HOME"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"

# Define the LLM and retriever models to evaluate here
# "ToolBench/ToolBench_IR_bert_based_uncased"
LLM_MODELS=("localg")
RETRIEVER_MODELS=("facebook/contriever-msmarco")

# Define the paths to the attacked results, B-Tools, and data here
ATTACKED_RESULTS_DIR="./attack_results"
B_TOOLs=("tool1" "tool2" "tool3")
DATA_PATH_TRAIN="./data/g1_train_a.json"
DATA_PATH_EVAL="./data/g1_eval_a.json"

# Define the assets to evaluate here
assets=("g1_train_a" "g1_train_b" "g1_train_c")
evals=("g1_eval_a" "g1_eval_b" "g1_eval_c")

nvidia-smi

for llm in "${LLM_MODELS[@]}"; do
    for retriever in "${RETRIEVER_MODELS[@]}"; do
        for i in "${!assets[@]}"; do
            asset="${assets[i]}"
            evaluation="${evals[i]}"
            b_tool="${B_TOOLs[i]}"
            retriever_name=$(echo $retriever | sed 's/\//_/g')

            result_files=("${ATTACKED_RESULTS_DIR}/${asset}/attack_results_${retriever_name}.json")

            for result_file in "${result_files[@]}"; do
                echo "Evaluating LLM: $llm, Retriever: $retriever, Result File: $result_file, B-Tool: $b_tool..."

                srun python evaluate.py \
                    --llm_model="$llm" \
                    --retriever_model="$retriever" \
                    --attacked_results="$result_file" \
                    --data_path_train="$DATA_PATH_TRAIN" \
                    --data_path_eval="$DATA_PATH_EVAL" \
                    --b_tool="$b_tool"
            done
        done
    done
done

echo "All evaluations completed!"