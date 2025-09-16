def prompts_read(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    return lines


def prompts_write(path):
    file = open(path, 'w')
    return file


path_file = "train/fucked"
lines = prompts_read("prompts")
file = prompts_write(path_file)

name = "fucked"
for i in range(len(lines)):
    prompt = lines[i]
    if name in prompt:
        file.write(prompt + '\n')
