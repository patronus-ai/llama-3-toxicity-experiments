# Turkish
python llama_experiments.py --num_samples=500 --output_folder="outputs" --label_col="is_toxic" --label_desired=1

# Thai
python llama_experiments.py --dataset_name="tmu-nlp/thai_toxicity_tweet" --text_col="tweet_text" --label_col="is_toxic" --num_samples=500 --output_folder="outputs"

# Ukrainian
python llama_experiments.py --dataset_name="ukr-detect/ukr-toxicity-dataset" --label_col="toxic" --num_samples=500 --output_folder="outputs"

# English - Toxicity Jigsaw
python llama_experiments.py --dataset_name="Arsive/toxicity_classification_jigsaw" --text_col="comment_text" --num_samples=500 --output_folder="outputs" --label_col="toxic" --label_desired=1

# German - Change path to local download of csv file
python llama_experiments.py --dataset_name="datasets/german-hate-speech-superset.csv" --text_col="labels" --num_samples=500 --output_folder="outputs" --label_desired=1 --label_col="labels"

# English toxic-text
python llama_experiments.py --dataset_name="nicholasKluge/toxic-text" --split="english" --text_col="toxic" --num_samples=500 --output_folder="outputs" --label_desired="#"

# Portuguese toxic-text
python llama_experiments.py --dataset_name="nicholasKluge/toxic-text" --split="portuguese" --text_col="toxic" --num_samples=500 --output_folder="outputs" --label_desired="#"

# LMSys Toxic Chat
python llama_experiments.py --dataset_name="lmsys/toxic-chat" --split="train" --text_col="user_input" --label_col="toxicity" --chat_response_col="model_output" --num_samples=-1 --data_config="toxicchat0124" --output_folder="outputs"

# BeaverTails
python llama_experiments.py --dataset_name="PKU-Alignment/BeaverTails" --split="30k_test" --text_col="prompt" --label_col="is_safe" --chat_response_col="response" --num_samples=500 --output_folder="outputs" --label_desired="False"

# Xstest-response
python llama_experiments.py --dataset_name="allenai/xstest-response" --split="train" --text_col="prompt" --label_col="label" --chat_response_col="response" --num_samples=-1 --label_desired="harmful" --split="response_harmfulness" --output_folder="outputs"

# Salad-Data
python llama_experiments.py --dataset_name="OpenSafetyLab/Salad-Data" --split="train" --text_col="augq" --num_samples=500 --data_config="attack_enhanced_set" --output_folder="outputs"
