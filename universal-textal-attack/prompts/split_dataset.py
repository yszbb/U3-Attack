import torch


def prompts_read(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    return lines


def prompts_write(path):
    file = open(path, 'w')
    return file


path_file = "naked"
lines = prompts_read(path_file)

file_train = prompts_write("train/naked")

file_test = prompts_write("test/naked")

num_test = 0

for i in range(len(lines)):
    prompt = lines[i]
    rand_num = torch.rand(1)
    if num_test < 48:
        if rand_num >= 0.55:
            num_test += 1
            file_test.write(prompt + '\n')
        else:
            file_train.write(prompt + '\n')
    else:
        file_train.write(prompt + '\n')
