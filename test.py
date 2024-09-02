from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import time

import torch

# print(torch.cuda.is_available())

'''
音色克隆
'''
# cosyvoice = CosyVoice('/mnt/bn/smart-customer-service/users/zitong/LLM/tts_model/CosyVoice-300M')
# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
# prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
# output = cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
# torchaudio.save('./all_result/zero_shot_example.wav', output['tts_speech'], 22050)

# output = cosyvoice.inference_zero_shot('你好，我是贝陪科技生成式语音大模型，请问有什么可以帮您的吗？', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
# torchaudio.save('./all_result/zero_shot_test.wav', output['tts_speech'], 22050)

# output = cosyvoice.inference_zero_shot('在面对挑战时，他展现了非凡的勇气与智慧。', '希望你以后能够做的比我还好呦。', prompt_speech_16k)
# torchaudio.save('./all_result/zero_shot_test2.wav', output['tts_speech'], 22050)



'''
指定音色，['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
'''
# cosyvoice = CosyVoice('/mnt/bn/smart-customer-service/users/zitong/LLM/tts_model/CosyVoice-300M-SFT')
# # sft usage
# print(cosyvoice.list_avaliable_spks())
# output = cosyvoice.inference_sft('你好，我是贝陪科技生成式语音大模型，请问有什么可以帮您的吗？', '中文男')
# torchaudio.save('./all_result/sft_zh_man.wav', output['tts_speech'], 22050)

# print(cosyvoice.list_avaliable_spks())
# output = cosyvoice.inference_sft('你好，我是贝陪科技生成式语音大模型，请问有什么可以帮您的吗？', '中文女')
# torchaudio.save('./all_result/sft_zh_woman.wav', output['tts_speech'], 22050)

# print(cosyvoice.list_avaliable_spks())
# output = cosyvoice.inference_sft('Hello, I am the Generative Speech Model of Beipei Technology. How can I help you?', '英文男')
# torchaudio.save('./all_result/sft_en_man.wav', output['tts_speech'], 22050)

# print(cosyvoice.list_avaliable_spks())
# output = cosyvoice.inference_sft('Hello, I am the Generative Speech Model of Beipei Technology. How can I help you?', '英文女')
# torchaudio.save('./all_result/sft_en_woman.wav', output['tts_speech'], 22050)



'''
增加情感语气，边笑边说<laughter></laughter> 换气<breath></breath> 重读<strong></strong>
'''
# cosyvoice = CosyVoice('/mnt/bn/smart-customer-service/users/zitong/LLM/tts_model/CosyVoice-300M-Instruct')
# instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
# output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<breath>勇气</breath>与<laughter>智慧</laughter>。', '中文女', '')
# torchaudio.save('breath_laughter_instruct.wav', output['tts_speech'], 22050)


# output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的勇气与智慧。', '中文女', 'he is very sad')
# torchaudio.save('test_sad.wav', output['tts_speech'], 22050)


# output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的勇气与智慧。', '中文女', 'he is very angry')
# torchaudio.save('test_angry.wav', output['tts_speech'], 22050)


# output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的勇气与智慧。', '中文女', 'Theo \'Crimson\',生气小女孩的语气')
# torchaudio.save('test_child.wav', output['tts_speech'], 22050)

# output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<laughter>智慧</laughter>。', '中文女', '')
# torchaudio.save('strong_laughter_instruct.wav', output['tts_speech'], 22050)

# output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<breath>勇气</breath>与<strong>智慧</strong>。', '中文女', '')
# torchaudio.save('breath_strong_instruct.wav', output['tts_speech'], 22050)


# output = cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文女', '')
# torchaudio.save('strong_strong_instruct.wav', output['tts_speech'], 22050)



'''
测试流式推理
'''
# cosyvoice = CosyVoice('/mnt/bn/smart-customer-service/users/zitong/LLM/tts_model/CosyVoice-300M-SFT')
# context = '''一只小蚂蚁在草丛里迷路了，它很害怕。这时，一只乌龟路过，问清了情况，决定帮助小蚂蚁找到回家的路。乌龟告诉小蚂蚁，要沿着一条小溪走，然后穿过一座大桥，就能看到它的家了。小蚂蚁非常感激乌龟，于是它俩一起出发去找家。路上，小蚂蚁在草丛里发现了一只睡觉的蝴蝶，它叫醒了蝴蝶，告诉它乌龟在帮助它找家。蝴蝶也加入了队伍，一起去找家。最后，他们顺利地找到了小蚂蚁的家，小蚂蚁的妈妈非常高兴，感激乌龟和蝴蝶的帮助。他们在一起吃了晚饭，然后乌龟和蝴蝶就离开了。这个故事告诉我们，帮助别人是一件很美好的事情，而且有时候我们也会从中得到收获和快乐。'''
# # context = '你好，我是贝陪科技生成式语音大模型，请问有什么可以帮您的吗？'
# # sft usage
# print(cosyvoice.list_avaliable_spks())
# # change stream=True for chunk stream inference
# # time1 = time
# for i, j in enumerate(cosyvoice.inference_sft(context, '中文女', stream=True)):
#     # 采样率
#     # print(j['tts_speech'].shape)
#     torchaudio.save('./stream_result/sft_{}_test2.wav'.format(i), j['tts_speech'], 22050)






# cosyvoice = CosyVoice('/mnt/bn/smart-customer-service/users/zitong/LLM/tts_model/CosyVoice-300M')
# # zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
# # prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
# prompt_speech_16k = load_wav('/mnt/bn/smart-customer-service/users/zitong/tts_project/CosyVoice/seed_2146.wav', 16000)
# # prompt_text = '希望你以后能够做的比我还好呦。'
# prompt_text = '今天早晨，市中心的主要道路因突发事故造成了严重堵塞。请驾驶员朋友们注意绕行，并听从现场交警的指挥。'

# # test_text = '从前，有一只可爱的小兔子，它住在一个美丽的森林里。小兔子有一身雪白的绒毛，就像云朵一样柔软。'
# text = '你知道为什么树叶会在秋天变黄吗？'
# text_list = ["你知道为什么树叶会在秋天变黄吗？", "如果你是队长，你会怎么带领你的团队赢得比赛？", "我们的树屋需要修缮，你有什么好主意吗？",
# "如果你有一个新朋友，你会怎么让他感到受欢迎？",
# "如果我们用树枝和叶子做一个房子，你会怎么设计它？"]
# for i in range(5):
# text_long = '从前，有一只可爱的小兔子，它住在一个美丽的森林里。小兔子有一身雪白的绒毛，就像云朵一样柔软。有一天，小兔子听说森林深处有一个神秘的花园，里面长满了各种各样神奇的花朵。于是，它决定去探险。小兔子蹦蹦跳跳地穿过茂密的树林，遇到了一只小松鼠。小松鼠告诉小兔子路上可能会有危险，但小兔子很勇敢，它没有害怕。继续往前走，小兔子看到了一条清澈的小溪。它小心地跳过小溪，又遇到了一只漂亮的蝴蝶。蝴蝶带着小兔子飞了一段路，终于找到了神秘花园。花园里的花朵五颜六色，美丽极了。小兔子在花园里开心地玩耍，忘记了时间。可是，天渐渐黑了，小兔子想起要回家。它带着美好的回忆，沿着来时的路往家走。这次冒险让小兔子变得更加勇敢和坚强。'


# print(cosyvoice.inference_zero_shot(text_long, prompt_text, prompt_speech_16k, stream=False))

# print(cosyvoice.inference_zero_shot(text_long, prompt_text, prompt_speech_16k, stream=True))



# for i in range(5):
    # 不进行预加载的耗时
# for idx in range(1):
#     audio_segments = []
#     for i, j in enumerate(cosyvoice.inference_zero_shot(text_long, prompt_text, prompt_speech_16k, stream=True)):
#     # for i, j in enumerate(cosyvoice.inference_zero_shot(text_list[idx], '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#         # torchaudio.save('./stream_result/zero_shot_{}_prompt.wav'.format(i), j['tts_speech'], 22050)
#         audio_segments.append(j['tts_speech'])
#     # full_audio = torch.cat(audio_segments, dim=1)
#     # # 保存完整的音频文件
#     # torchaudio.save('single_result_man/complete_audio_stream_test_woman_long{}.wav'.format(idx), full_audio, 22050)
#     # print("Audio file has been saved.")

# 进行预加载时的耗时
# for idx in range(2):
#     audio_segments = []
#     for i, j in enumerate(cosyvoice.inference_zero_shot_preprocess(text_long, prompt_text, prompt_speech_16k, stream=True)):
#     # for i, j in enumerate(cosyvoice.inference_zero_shot(text_list[idx], '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#         # torchaudio.save('./stream_result/zero_shot_{}_prompt.wav'.format(i), j['tts_speech'], 22050)
#         audio_segments.append(j['tts_speech'])
#     # full_audio = torch.cat(audio_segments, dim=1)
#     # # 保存完整的音频文件
#     # torchaudio.save('single_result_man/complete_audio_stream_test_woman_long{}.wav'.format(idx), full_audio, 22050)
#     # print("Audio file has been saved.")



# for idx in range(5):
#     audio_segments = []
#     for i, j in enumerate(cosyvoice.inference_zero_shot_preprocess(text, prompt_text, prompt_speech_16k, stream=True)):
#     # for i, j in enumerate(cosyvoice.inference_zero_shot(text_list[idx], '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#         # torchaudio.save('./stream_result/zero_shot_{}_prompt.wav'.format(i), j['tts_speech'], 22050)
#         audio_segments.append(j['tts_speech'])
#     # full_audio = torch.cat(audio_segments, dim=1)
#     # # 保存完整的音频文件
#     # torchaudio.save('single_result_man/complete_audio_stream_test_woman_long{}.wav'.format(idx), full_audio, 22050)
#     # print("Audio file has been saved.")