import subprocess
import json
import warnings

def test(): 
    warnings.filterwarnings("ignore")

    python_command_1 = "python train.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path /home/jeff_lab/fyp/dnabert/finetune_ft1/ft_data \
    --kmer -1 \
    --run_name DNABERT2_/home/jeff_lab/fyp/dnabert/finetune_ft1/ft_data \
    --model_max_length 20 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --fp16 \
    --save_steps 15 \
    --save_model True \
    --output_dir output/dnabert2_10 \
    --evaluation_strategy steps \
    --eval_steps 15 \
    --warmup_steps 50 \
    --logging_steps 50 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False"

    python_command_2 = "python ft1dnabert.py"

    with open('output_10.txt', 'a') as f:
        # Repeat the process 50 times
        for i in range(50):
            # Run the first Python command and capture the output
            process = subprocess.Popen(python_command_1.split(), stdout=subprocess.PIPE)
            output1, _ = process.communicate()

            # Save the output to the text file
            f.write(f"Iteration {i + 1} - Command 1 Output:\n")
            f.write(output1.decode('utf-8'))
            f.write("\n")

            # Run the second Python command
            process = subprocess.Popen(python_command_2.split(), stdout=subprocess.PIPE)
            output2, _ = process.communicate()

            # Save the output to the text file
            f.write(f"Iteration {i + 1} - Command 2 Output:\n")
            f.write(output2.decode('utf-8'))
            f.write("\n")

if __name__ == "__main__":
    test()