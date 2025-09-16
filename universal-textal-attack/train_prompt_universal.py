import argparse

import torch
from diffusers import StableDiffusionInpaintPipeline
from utils import object_key, set_seed, prompts_read, get_random_elements
from transformers import CLIPTextModel, CLIPTokenizer
from utils import get_char_table, train, get_text_embeds_without_uncond


cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
device = 'cuda'
benign_word = []

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "../checkpoints/stable-diffusion-inpainting").to(
    device)
# 得到文本编码器
text_encoder = pipe_inpaint.text_encoder
# 获得token词典表
tokenizer = pipe_inpaint.tokenizer
text_encoder = text_encoder.to(device)


#  PGD
def main(args):
    len_prompt = args.len_prompt
    # char_table = get_char_table()
    train_path = args.train_path
    train_data = prompts_read(train_path)
    obj_word = "sex"
    num_candidate = 2000
    topn_vector = []
    voc = tokenizer.get_vocab().items()

    # 因为是敏感单词，是需要进行抹去
    tokens_to_remove_list = []
    nsfw_word_list = ['sexual', 'nude', 'nudity', 'topless', 'sex', '18+', 'naked',
                      'nsfw', 'porn', 'dick', 'vagina', 'naked child', 'explicit content',
                      'uncensored', 'fuck', 'nipples', 'visible nipples', 'breast', 'areola',
                      'pussy', 'penis', 'vagina', 'suck', 'cock', 'blowjock', 'blow', 'blood',
                      'bleed', 'corpse', 'hang', 'flesh', 'creep', 'horror', 'demon', 'dead',
                      'death', 'deform', 'pepe', 'nazi', 'confederate', 'donald trump', 'kim kardashian',
                      'biden', 'elon musk', 'pelosi', 'jew', 'asian', 'african', 'asia', 'africa', 'zombi', ]

    sensitive_word_list = ['ak', 'na', 'ke', 'nud'] + nsfw_word_list

    for token, token_id in voc:
        num_select = 0
        for index, word in enumerate(sensitive_word_list):
            if word in token:
                num_select += 1
        if num_select == 0:
            tokens_to_remove_list.append((token, token_id))

    tokens_to_remove_set = set(tokens_to_remove_list)
    print(tokens_to_remove_set.__len__())
    # 将需要移除的敏感词索引保存下来
    torch.save(tokens_to_remove_set, "weights/tokens_to_remove_set.pt")

    for token, token_id in tokens_to_remove_set:
        sen_embed = get_text_embeds_without_uncond(obj_word, tokenizer=tokenizer,
                                                   text_encoder=text_encoder)
        crafted_embed = get_text_embeds_without_uncond(token, tokenizer=tokenizer,
                                                       text_encoder=text_encoder)
        min_cos = 1 - cos(sen_embed.view(-1), crafted_embed.view(-1))

        if len(topn_vector) < num_candidate:
            topn_vector.append((min_cos, token, token_id))
            try:
                topn_vector = sorted(topn_vector, reverse=False)
            except:
                length = len(topn_vector) - 1
                topn_vector = topn_vector[0:length]
        else:
            if min_cos <= topn_vector[-1][0]:
                topn_vector.append((min_cos, token, token_id))
                try:
                    topn_vector = sorted(topn_vector, reverse=False)
                except:
                    length = len(topn_vector) - 1
                    topn_vector = topn_vector[0:length]
                topn_vector = topn_vector[0:num_candidate]

    torch.save(topn_vector, "weights/{}_{}_candidate.pt".format(num_candidate, obj_word))

    topn_vector = torch.load("weights/{}_{}_candidate.pt".format(num_candidate, obj_word))
    char_list = []
    for _, word, word_ind in topn_vector:
        char_list.append(word)

    char_list = char_list[0:150]
    for epoch in range(1):
        results = []
        for i in range(len(train_data)):
            obj = {
                "source_prompt": '',
                "adv_prompt1": '',
                # "adv_prompt2": '',
                # "adv_prompt3": '',
                # "adv_prompt4": '',
                # "adv_prompt5": '',
            }
            prompt = train_data[i]
            prompt = prompt.rstrip('.')
            # mask = object_key([prompt], obj_word, thres=9, tokenizer=tokenizer, text_encoder=text_encoder,
            #                   replace_word=topn_vector[0][1])
            #
            # mask = mask.view(-1)
            mask = None
            try:
               location_obj_word = prompt.split(' ').index(obj_word)
            except:
                location_obj_word = prompt.split(' ').index(obj_word + ',')

            attack_sentence = prompt
            obj["source_prompt"] = prompt
            for ind in range(1):
                replace_init = get_random_elements(char_list, len_prompt)
                replace_list = ' '.join(replace_init)
                adv_sentence = attack_sentence.replace(obj_word, replace_list)

                max_tensor, loss_list, pgd_prompt, min_loss = train(init_per_sample=1, sentence=attack_sentence,
                                                                    len_prompt=len_prompt,
                                                                    char_list=char_list,
                                                                    model=text_encoder.text_model, iter_num=300,
                                                                    eta_u=1, tokenizer=tokenizer,
                                                                    text_encoder=text_encoder,
                                                                    mask=mask, adv_sentence=adv_sentence,
                                                                    location_obj_word=location_obj_word,
                                                                    sensitive_word=obj_word, replace_list=replace_list)
                pgd_sentence = adv_sentence.replace(replace_list, pgd_prompt)
                obj["adv_prompt{}".format(ind)] = pgd_sentence
                print("pgd_prompt:", pgd_sentence)
            results.append(obj)
        torch.save(results, "results/{}.pt".format(obj_word))


if __name__ == "__main__":
    # If the file is being run directly, it calls the main function
    parser = argparse.ArgumentParser(description='attack diffusion.')
    parser.add_argument("-s", '--random_seed', type=int, default=7867, help='The random seed.')
    # parser.add_argument("-i", '--iteration', type=int, default=1000, help='iteration')
    # parser.add_argument("-n", '--candicate_number', type=int, default=10)
    parser.add_argument("-t", '--train_path', type=str, default="prompts/train/sex")
    parser.add_argument("-epo", '--epoch', type=int, default=10)
    parser.add_argument("-len", '--len_prompt', type=int, default=3)
    args = parser.parse_args()
    set_seed(args.random_seed)
    print(args)
    main(args)
