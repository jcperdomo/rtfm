
: ${MAX_SAMPLES:=100}
: ${SERIALIZER_CLS:="BasicSerializerV2"}
: ${CONTEXT_LENGTH:="8192"}
# : ${SHOT_SEL:="random"}


# set CKPT_DIR to the the hf model name string in order to use the base model 
# (e.g. "meta-llama/Meta-Llama-3-8B") or set it to a local directory
# containing a model in HF format.
: ${CKPT_DIR:="mlfoundations/tabula-8b"}

# The executable used to run evals.
: ${EVAL_SCRIPT:="sbatch/evaluate_no_accelerate_jsc_v2.sbatch"}

echo "##############################################"
echo "SERIALIZER_CLS is ${SERIALIZER_CLS}"
echo "MAX_SAMPLES is ${MAX_SAMPLES}"
echo "CKPT_DIR is ${CKPT_DIR}"
echo "##############################################"


# : ${TASK:="grinsztajn/cat_clf/albert"}


# for TASK in evaldatasets/grinsztajn/*/*; do
# for TASK in evaldatasets/grinsztajn/*/*; do
for TASK in evaldatasets/openml_cc18/*; do
    TASK=${TASK#evaldatasets/} 
# [ -d "${TASK}" ] && 
    for SHOT_SEL in "random" "rices"; do
        echo "Task is ${TASK}"
        echo "SHOT SEL is ${SHOT_SEL}"
        sbatch --export=TASK=${TASK},SHOT_SEL=${SHOT_SEL},CKPT_DIR=${CKPT_DIR},SERIALIZER_CLS=${SERIALIZER_CLS},MAX_SAMPLES=${MAX_SAMPLES},CONTEXT_LENGTH=${CONTEXT_LENGTH} --job-name=${TASK} ${EVAL_SCRIPT}
    done
done

# AMLB
# for TASK in data_scientist_salary imdb_genre_prediction jigsaw_unintended_bias100K kick_starter_funding melbourne_airbnb news_channel product_sentiment_machine_hack wine_reviews; do
#   sbatch --export=TASK=${TASK},CKPT_DIR=${CKPT_DIR},SERIALIZER_CLS=${SERIALIZER_CLS},MAX_SAMPLES=${MAX_SAMPLES},CONTEXT_LENGTH=${CONTEXT_LENGTH} --job-name=${TASK} ${EVAL_SCRIPT}
# done

# Catboost

#for TASK in amazon kick adult appetency churn click upselling; do
#sbatch --export=TASK=${TASK},CKPT_DIR="/gscratch/efml/jpgard/tablm/checkpoints/v6.0.3-lr1e-5-BasicSerializerV2-gbs24-steps8_k-Meta-Llama-3-8B/",SERIALIZER_CLS=${SERIALIZER_CLS},MAX_SAMPLES=${MAX_SAMPLES},CONTEXT_LENGTH=${CONTEXT_LENGTH} --job-name=${TASK} sbatch/eval/evaluate_no_accelerate_hyak_cpu.sbatch
#done


# Grinsztajn
# for TASK in evaldatasets/grinsztajn/*/*; do
# [ -d "${TASK}" ] && sbatch --export=TASK=${TASK},CKPT_DIR=${CKPT_DIR},SERIALIZER_CLS=${SERIALIZER_CLS},MAX_SAMPLES=${MAX_SAMPLES},CONTEXT_LENGTH=${CONTEXT_LENGTH} --job-name=${TASK} ${EVAL_SCRIPT}
# done

# Unipredict
# while IFS= read -r TASK; do
#   sbatch --export=TASK=${TASK},CKPT_DIR=${CKPT_DIR},SERIALIZER_CLS=${SERIALIZER_CLS},MAX_SAMPLES=${MAX_SAMPLES},CONTEXT_LENGTH=${CONTEXT_LENGTH} --job-name=${TASK} ${EVAL_SCRIPT}
# done < "unipredict_supervised_tasks.txt"

# OpenML-CC18
# for TASK in tmp/openml_cc18/*; do
# [ -d "${TASK}" ] && sbatch --export=TASK=${TASK#tmp/},CKPT_DIR=${CKPT_DIR},SERIALIZER_CLS=${SERIALIZER_CLS},MAX_SAMPLES=${MAX_SAMPLES},CONTEXT_LENGTH=${CONTEXT_LENGTH} --job-name=${TASK} ${EVAL_SCRIPT}
# done

# OpenML-CTR23
# for TASK in tmp/openml_ctr23/*; do
# [ -d "${TASK}" ] && sbatch --export=TASK=${TASK#tmp/},CKPT_DIR=${CKPT_DIR},SERIALIZER_CLS=${SERIALIZER_CLS},MAX_SAMPLES=${MAX_SAMPLES},CONTEXT_LENGTH=${CONTEXT_LENGTH} --job-name=${TASK} ${EVAL_SCRIPT}
# done
