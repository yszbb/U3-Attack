import logging

import torch
import copy
import random
from torch.nn import functional as F
import numpy as np
from PIL import Image

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)


def get_text_embeds_without_uncond(prompt, tokenizer, text_encoder):
    # Tokenize text and get embeddings
    # 字典长度::::::::::::::::::49408
    # 1 * 77
    text_input = tokenizer(
        prompt, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt')
    with torch.no_grad():
        # 1 * 77 * 768  最终输出
        text_embeddings = text_encoder(text_input.input_ids.cuda())[0]
        # text_embeddings = text_encoder(text_input.input_ids)[0]
    return text_embeddings


def get_char_table():
    char_table = ['·', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '=', '-', '*', '+', '.', '<', '>', '?',
                  ',', '\'', ';', ':', '|', '\\', '/']
    for i in range(ord('A'), ord('Z') + 1):
        char_table.append(chr(i))
    for i in range(0, 10):
        char_table.append(str(i))
    return char_table


# greedy algorithm
def cos_mask(a, b, mask):
    return cos(a * mask, b * mask)


# PGD
def get_clip_embedding(prompt, tokenizer=None, text_encoder=None):
    text_input = tokenizer(
        prompt, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt')

    input_ids = text_input.input_ids
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    input_ids = input_ids.cuda()
    with torch.no_grad():
        txt_embed = text_encoder.text_model.embeddings(input_ids=input_ids)
    return txt_embed, input_shape


def forward_embedding(hidden_states, input_shape, model=None, tokenizer=None, text_encoder=None):
    output_attentions = text_encoder.text_model.config.output_attentions
    output_hidden_states = (
        text_encoder.text_model.config.output_hidden_states
    )
    bsz, seq_len = input_shape
    return_dict = text_encoder.text_model.config.use_return_dict
    causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len).to(hidden_states.device)
    attention_mask = None
    encoder_outputs = text_encoder.text_model.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    last_hidden_state = encoder_outputs[0]
    last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)
    return last_hidden_state


def forward_embedding_no_grad(hidden_states, input_shape, model=None, tokenizer=None, text_encoder=None):
    with torch.no_grad():
        output_attentions = text_encoder.text_model.config.output_attentions
        output_hidden_states = (
            text_encoder.text_model.config.output_hidden_states
        )
        bsz, seq_len = input_shape
        return_dict = text_encoder.text_model.config.use_return_dict

        causal_attention_mask = text_encoder.text_model._build_causal_attention_mask(bsz, seq_len).to(
            hidden_states.device)
        attention_mask = None
        encoder_outputs = text_encoder.text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = text_encoder.text_model.final_layer_norm(last_hidden_state)
    return last_hidden_state


class PGDattack():
    def __init__(self):
        self.stdste = True
        # self.stdste = False

    def project_u_tensor(self, u_tensor, site_mask, sub_mask):
        skip = site_mask == 0
        subword_opt = sub_mask != 0
        for i in range(u_tensor.size(0)):
            if skip[i]:
                continue
            # u_tensor[i][subword_opt[i]] = u_tensor[i][subword_opt[i]] + self.eta_u * u_grad[i][subword_opt[i]]
            # print("before project: ", u_tensor[i][subword_opt[i]])
            u_tensor[i][subword_opt[i]] = self.bisection_u(u_tensor[i][subword_opt[i]], eps=1)
            # print("after project: ", u_tensor[i][subword_opt[i]])
            # print(torch.abs(torch.sum(u_tensor[i][subword_opt[i]]) - 1))
            assert torch.abs(torch.sum(u_tensor[i][subword_opt[i]]) - 1) <= 1e-3
        return u_tensor

    def bisection_u(self, a, eps, xi=1e-5, ub=1):
        pa = torch.clip(a, 0, ub)
        if np.abs(torch.sum(pa).item() - eps) <= xi:
            # print('np.sum(pa) <= eps !!!!')
            upper_S_update = pa
        else:
            mu_l = torch.min(a - 1).item()
            mu_u = torch.max(a).item()
            # mu_a = (mu_u + mu_l)/2
            while np.abs(mu_u - mu_l) > xi:
                # print('|mu_u - mu_l|:',np.abs(mu_u - mu_l))
                mu_a = (mu_u + mu_l) / 2
                gu = torch.sum(torch.clip(a - mu_a, 0, ub)) - eps
                gu_l = torch.sum(torch.clip(a - mu_l, 0, ub)) - eps + 1e-8
                gu_u = torch.sum(torch.clip(a - mu_u, 0, ub)) - eps
                # print('gu:',gu)
                if gu == 0:
                    # print('gu == 0 !!!!!')
                    break
                elif gu_l == 0:
                    mu_a = mu_l
                    break
                elif gu_u == 0:
                    mu_a = mu_u
                    break
                # if torch.sign(gu) == torch.sign(gu_l):
                #     mu_l = mu_a
                # else:
                #     mu_u = mu_a
                if gu * gu_l < 0:  ## 右侧大于0，中值小于0
                    mu_l = mu_l
                    mu_u = mu_a
                elif gu * gu_u < 0:  ## 左侧小于0，中值大于0
                    mu_u = mu_u
                    mu_l = mu_a
                else:
                    print(a)
                    print(gu, gu_l, gu_u)
                    raise Exception()

            upper_S_update = torch.clip(a - mu_a, 0, ub)

        return upper_S_update

    def estimate_u_tensor(self, u_tensor, site_mask, ):
        if self.stdste:
            return F.gumbel_softmax(u_tensor, tau=0.5, hard=True, dim=-1)
            # return F.gumbel_softmax(u_tensor, tau = 0.1, hard = False, dim = -1)
        for i in range(u_tensor.size(0)):
            if site_mask[i] == 0:
                continue
            # u_tensor[i] = STDSTERandSelect.apply(u_tensor[i])
            # print(u_tensor[i])
            u_tensor[i] = STERandSelect.apply(u_tensor[i])
        return u_tensor


class STERandSelect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        prob = input.cpu().detach().numpy()
        prob = prob / np.sum(prob)
        substitute_idx = np.random.choice(input.size(0), p=prob)
        new_vector = torch.zeros_like(input).to(input.device)
        new_vector[substitute_idx] = 1
        return new_vector

    @staticmethod
    def backward(ctx, grad_output):
        # return grad_output
        return F.hardtanh(grad_output)


def craft_candidate_embed(char_list, tokenizer=None, text_encoder=None):
    prompt_char = ''
    position_length = 75
    candidate_embedding = []
    for i in char_list:
        prompt_char = ''
        for position in range(position_length):
            prompt_char += i
            prompt_char += ' '
        char_posit_embed = get_clip_embedding(prompt_char, tokenizer=tokenizer, text_encoder=text_encoder)[0]
        candidate_embedding.append(char_posit_embed)
    candidate_embedding = torch.stack(candidate_embedding, dim=1)
    candidate_embedding = candidate_embedding.permute(0, 2, 1, 3)
    return candidate_embedding


def train(init_per_sample, sentence, len_prompt, char_list, model, iter_num=500, eta_u=1, mask=None, tokenizer=None,
          text_encoder=None, adv_sentence=None, location_obj_word=None, sensitive_word=None, replace_list=None):
    num_word = len(adv_sentence.split(' '))
    seq_len = 77
    neighbor_num = len(char_list)

    probability_op = torch.zeros([init_per_sample, len_prompt, neighbor_num], dtype=torch.double).fill_(
        1 / neighbor_num)
    probability_op.requires_grad = True
    u_tensor = torch.zeros([init_per_sample, seq_len, neighbor_num], dtype=torch.double).fill_(1 / neighbor_num)
    u_tensor[:, location_obj_word + 1:location_obj_word + len_prompt + 1, :] = probability_op

    fixed_z = torch.zeros(init_per_sample, seq_len)
    fixed_z[0][location_obj_word + 1:location_obj_word + len_prompt + 1] = 1

    # fixed_replace = torch.zeros(init_per_sample, seq_len)
    # fixed_replace[0][1:len_prompt + 1] = 1

    # 制作候选编码 1 * 77 * neighbor_num * 768
    candidate_embed = craft_candidate_embed(char_list, tokenizer=tokenizer, text_encoder=text_encoder)

    orig_adv_embeddings, input_adv_shape = get_clip_embedding(adv_sentence, tokenizer=tokenizer,
                                                              text_encoder=text_encoder)

    orig_output = get_text_embeds_without_uncond(sentence, tokenizer=tokenizer,
                                                 text_encoder=text_encoder)

    replace_list_embeddings, _ = get_clip_embedding(replace_list, tokenizer=tokenizer,
                                                    text_encoder=text_encoder)
    orig_sensitive_word_output = get_text_embeds_without_uncond(sensitive_word, tokenizer=tokenizer,
                                                                text_encoder=text_encoder)

    model.train()
    loss_list = []
    single_loss_list = []
    batch_size = 100
    sign = True
    simplex = False
    bool_max = False  # maximize loss or minimize loss
    n = 5  # top n for backward
    pgd = PGDattack()
    for i in range(iter_num):
        topn_vector = []  # {loss, vector}
        for j in range(batch_size):
            discrete_u = pgd.estimate_u_tensor(u_tensor, fixed_z.view(-1))
            # discrete_u=u_tensor  # 1 * 77 * 63 * 1
            discrete_u = discrete_u.view(init_per_sample, seq_len, neighbor_num, 1)
            discrete_u = discrete_u.cuda()
            # 1 * 77 * neighbor_num(63) * 768
            subword_embeddings = candidate_embed.view(1, seq_len, neighbor_num, -1)
            # 1 * 77 * 1
            discrete_z = fixed_z.view(init_per_sample, seq_len, 1)
            discrete_z = discrete_z.cuda()
            # 1 * 77 * 768
            orig_adv_embeddings = orig_adv_embeddings.view(1, seq_len, -1)
            # 对抗编码
            new_embeddings = (1 - discrete_z) * orig_adv_embeddings + discrete_z * torch.sum(
                discrete_u * subword_embeddings, dim=2)
            new_embeddings = new_embeddings.float()

            new_output = forward_embedding_no_grad(new_embeddings, [1, 77], model=model, tokenizer=tokenizer,
                                                   text_encoder=text_encoder)

            if mask is not None:
                temp_loss_1 = 1 - cos_mask(new_output.view(-1), orig_output.view(-1), mask)
            else:
                temp_loss_1 = 1 - cos(new_output.view(-1), orig_output.view(-1))

            # new_embeddings_replace = discrete_u[:, location_obj_word + 1:location_obj_word + len_prompt + 1, :,
            #                          :] * subword_embeddings[:, 1:len_prompt + 1, :, :]
            # new_embeddings_sum = torch.sum(new_embeddings_replace, dim=2)
            # replace_list_embeddings[:, 1:len_prompt + 1, :] = new_embeddings_sum.float()
            # new_output_replace = forward_embedding_no_grad(replace_list_embeddings, [1, 77], model=model,
            #                                                tokenizer=tokenizer,
            #                                                text_encoder=text_encoder)
            #
            # temp_loss_2 = 1 - cos(new_output_replace.view(-1), orig_sensitive_word_output.view(-1))
            temp_loss_2 = 0
            temp_loss = 1.0 * temp_loss_1 + 0.5 * temp_loss_2

            single_loss_list.append(temp_loss.item())

            if len(topn_vector) < n:
                # 这个地方是干嘛的
                topn_vector.append((temp_loss.item(), discrete_u))
                try:
                    topn_vector = sorted(topn_vector, reverse=False)
                except:
                    length = len(topn_vector) - 1
                    topn_vector = topn_vector[0:length]
            else:
                if temp_loss.item() <= topn_vector[-1][0]:
                    topn_vector.append((temp_loss.item(), discrete_u))
                    try:
                        topn_vector = sorted(topn_vector, reverse=False)
                    except:
                        length = len(topn_vector) - 1
                        topn_vector = topn_vector[0:length]
                    topn_vector = topn_vector[0:n]

            if temp_loss.item() <= min(single_loss_list):
                min_loss = temp_loss.item()
                print("min loss is::::", min_loss)
                # print(temp_loss_1, temp_loss_2)
                min_tensor = discrete_u
                res_list = min_tensor.view(77, -1)[num_word + 1:num_word + len_prompt + 1].argmax(dim=1)
        # ################# 获得排名前五的结果 ####################

        total_loss = 0
        for k in range(len(topn_vector)):
            max_vector = topn_vector[k][1]
            new_embeddings = (1 - discrete_z) * orig_adv_embeddings + discrete_z * torch.sum(
                max_vector * subword_embeddings, dim=2)

            new_embeddings = new_embeddings.float()
            new_output = forward_embedding(new_embeddings, [1, 77], model, tokenizer=tokenizer,
                                           text_encoder=text_encoder)
            if mask is not None:
                loss_1 = 1 - cos_mask(new_output.view(-1), orig_output.view(-1), mask)
            else:
                loss_1 = 1 - cos(new_output.view(-1), orig_output.view(-1))
            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # new_embeddings_replace = max_vector[:, location_obj_word + 1:location_obj_word + len_prompt + 1, :,
            #                          :] * subword_embeddings[:, 1:len_prompt + 1, :, :]
            # new_embeddings_sum = torch.sum(new_embeddings_replace, dim=2)
            # replace_list_embeddings[:, 1:len_prompt + 1, :] = new_embeddings_sum.float()
            # new_output_replace = forward_embedding(replace_list_embeddings, [1, 77], model=model,
            #                                        tokenizer=tokenizer,
            #                                        text_encoder=text_encoder)
            #
            # loss_2 = 1 - cos(new_output_replace.view(-1), orig_sensitive_word_output.view(-1))
            loss_2 = 0.
            total_loss += (1.0 * loss_1 + 0.5 * loss_2)

        loss_list.append(total_loss.item() / n)

        # loss_mean = total_loss / n
        total_loss.backward(retain_graph=True)
        lr = eta_u / np.sqrt(iter_num)

        probability_op_grad = probability_op.grad

        if sign:
            probability_op_grad = torch.sign(probability_op_grad)

        probability_op_clone = probability_op.detach().clone()
        probability_op_update = lr * probability_op_grad
        if bool_max:
            probability_op_opt = probability_op_clone + probability_op_update
        else:
            probability_op_opt = probability_op_clone - probability_op_update

        # sub_mask = torch.ones(u_tensor_opt.shape)
        # u_tensor_shape = u_tensor_opt.shape

        # if simplex:
        #     u_tensor_opt = pgd.project_u_tensor(u_tensor_opt[0], fixed_z.view(-1), sub_mask[0])

        # u_tensor_opt = u_tensor_opt.view(u_tensor_shape)
        probability_op.data = probability_op_opt
        probability_op.grad.zero_()
        u_tensor[:, location_obj_word + 1:location_obj_word + len_prompt + 1, :] = probability_op

    res_list = min_tensor.view(77, -1)[location_obj_word + 1:location_obj_word + len_prompt + 1].argmax(dim=1)
    # print(''.join([char_list[x] for x in res_list]))
    return min_tensor, loss_list, ' '.join([char_list[x] for x in res_list]), min(single_loss_list)


def object_key(sentence_list, object_word, thres=10, tokenizer=None, text_encoder=None, replace_word=None):
    extra_words = object_word
    diff_list = []
    total_diff = 0
    for i in sentence_list:
        # i = "A group of naked women and men posing for the camera"
        sen_embed = get_text_embeds_without_uncond(i, tokenizer=tokenizer, text_encoder=text_encoder)
        crafted_embed = get_text_embeds_without_uncond(i.replace(object_word, replace_word), tokenizer=tokenizer,
                                                       text_encoder=text_encoder)

        diff_list.append(crafted_embed - sen_embed)
    #     total_diff += crafted_embed - sen_embed
    # average_diff = total_diff / len(diff_list)

    total_sign = 0
    for vec in diff_list:
        vec[abs(vec) > 0] = 1
        vec[vec == 0] = 0
        total_sign = vec

    # for vec in diff_list:
    #     vec[vec > 0] = 1
    #     total_sign += vec
    # total_sign[abs(total_sign) <= thres] = 0
    # total_sign[abs(total_sign) > thres] = 1
    # print('Ratio of mask', total_sign[total_sign > 0].shape[0] / total_sign.view(-1).shape[0])
    return total_sign


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prompts_read(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    return lines


def get_random_elements(my_list, N):
    random_elements = random.sample(my_list, N)
    return random_elements
