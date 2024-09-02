# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import numpy as np
import threading
import time
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out


class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.token_min_hop_len = 100
        self.token_max_hop_len = 200
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = 34
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.flow_hift_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device))
        self.llm.to(self.device).eval()
        self.llm.half()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device))
        self.flow.to(self.device).eval()
        self.hift.load_state_dict(torch.load(hift_model, map_location=self.device))
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model)
        self.llm.llm = llm_llm

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context:
            for i in self.llm.inference(text=text.to(self.device),
                                                text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                embedding=llm_embedding.to(self.device).half(),
                                                sampling=25,
                                                max_token_text_ratio=30,
                                                min_token_text_ratio=3):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True


    def llm_job_v2(self, text, text_len, prompt_text, prompt_text_len, llm_prompt_speech_token, llm_prompt_speech_token_len, llm_embedding):
        with self.llm_context:
            for i in self.llm.inference_v2(text=text.to(self.device),
                                                text_len=text_len.to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=prompt_text_len.to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
                                                embedding=llm_embedding.to(self.device),
                                                beam_size=1,
                                                sampling=25,
                                                max_token_text_ratio=30,
                                                min_token_text_ratio=3,
                                                stream=True):
                self.tts_speech_token.append(i)
        self.llm_end = True


    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False):
        with self.flow_hift_context:
            tts_mel = self.flow.inference(token=token.to(self.device),
                                        token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_token=prompt_token.to(self.device),
                                        prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                        prompt_feat=prompt_feat.to(self.device),
                                        prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                        embedding=embedding.to(self.device))
            # mel overlap fade in out
            if self.mel_overlap_dict[uuid] is not None:
                tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
            # append hift cache
            if self.hift_cache_dict[uuid] is not None:
                hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
            else:
                hift_cache_source = torch.zeros(1, 1, 0)
            # keep overlap mel and hift cache
            if finalize is False:
                self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
                tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
                tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
                self.hift_cache_dict[uuid] = {'source': tts_source[:, :, -self.source_cache_len:], 'mel': tts_mel[:, :, -self.mel_cache_len:]}
                tts_speech = tts_speech[:, :-self.source_cache_len]
            else:
                tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
        return tts_speech

    def inference(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                  llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                  flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid], self.mel_overlap_dict[this_uuid], self.hift_cache_dict[this_uuid] = [], False, None, None
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.concat(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len], dim=1)
                    with self.flow_hift_context:
                        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                    prompt_token=flow_prompt_speech_token,
                                                    prompt_feat=prompt_speech_feat,
                                                    embedding=flow_embedding,
                                                    uuid=this_uuid,
                                                    finalize=False)
                    yield  {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.concat(self.tts_speech_token_dict[this_uuid], dim=1)
            with self.flow_hift_context:
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                            prompt_token=flow_prompt_speech_token,
                                            prompt_feat=prompt_speech_feat,
                                            embedding=flow_embedding,
                                            uuid=this_uuid,
                                            finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.concat(self.tts_speech_token_dict[this_uuid], dim=1)
            with self.flow_hift_context:
                this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                            prompt_token=flow_prompt_speech_token,
                                            prompt_feat=prompt_speech_feat,
                                            embedding=flow_embedding,
                                            uuid=this_uuid,
                                            finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        torch.cuda.synchronize()




    def inference_v2(self, text, text_len, flow_embedding, llm_embedding=torch.zeros(0, 192),
                  prompt_text=torch.zeros(1, 0, dtype=torch.int32), prompt_text_len=torch.zeros(1, dtype=torch.int32),
                  llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), llm_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32), flow_prompt_speech_token_len=torch.zeros(1, dtype=torch.int32),
                  prompt_speech_feat=torch.zeros(1, 0, 80), prompt_speech_feat_len=torch.zeros(1, dtype=torch.int32), stream=False):
        if stream is True:
            self.tts_speech_token, self.llm_end, cache_speech = [], False, None
            p = threading.Thread(target=self.llm_job_v2, args=(text.to(self.device), text_len.to(self.device), prompt_text.to(self.device), prompt_text_len.to(self.device),
                                                     llm_prompt_speech_token.to(self.device), llm_prompt_speech_token_len.to(self.device), llm_embedding.to(self.device)))
            self.stream_win_len = 60 * 4
            self.stream_hop_len = 50 * 4
            self.overlap = 4395 * 4
            p.start()
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token) >= self.stream_win_len:
                    this_tts_speech_token = torch.concat(self.tts_speech_token[:self.stream_win_len], dim=1)
                    with self.flow_hift_context:
                        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                    prompt_token=flow_prompt_speech_token.to(self.device),
                                                    prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                                    prompt_feat=prompt_speech_feat.to(self.device),
                                                    prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                                    embedding=flow_embedding.to(self.device))
                    # fade in/out if necessary
                    if cache_speech is not None:
                        this_tts_speech[:, :self.overlap] = this_tts_speech[:, :self.overlap] * self.window[:self.overlap] + cache_speech * self.window[-self.overlap:]
                    yield  {'tts_speech': this_tts_speech[:, :-self.overlap]}
                    cache_speech = this_tts_speech[:, -self.overlap:]
                    with self.lock:
                        self.tts_speech_token = self.tts_speech_token[self.stream_hop_len:]
                if self.llm_end is True:
                    break
            # deal with remain tokens
            if cache_speech is None or len(self.tts_speech_token) > self.stream_win_len - self.stream_hop_len:
                this_tts_speech_token = torch.concat(self.tts_speech_token, dim=1)
                with self.flow_hift_context:
                    this_tts_mel = self.flow.inference(token=this_tts_speech_token,
                                                token_len=torch.tensor([this_tts_speech_token.size(1)], dtype=torch.int32).to(self.device),
                                                prompt_token=flow_prompt_speech_token.to(self.device),
                                                prompt_token_len=flow_prompt_speech_token_len.to(self.device),
                                                prompt_feat=prompt_speech_feat.to(self.device),
                                                prompt_feat_len=prompt_speech_feat_len.to(self.device),
                                                embedding=flow_embedding.to(self.device))
                    this_tts_speech = self.hift.inference(mel=this_tts_mel).cpu()
                if cache_speech is not None:
                    this_tts_speech[:, :self.overlap] = this_tts_speech[:, :self.overlap] * self.window[:self.overlap] + cache_speech * self.window[-self.overlap:]
                yield {'tts_speech': this_tts_speech}
            else:
                assert len(self.tts_speech_token) == self.stream_win_len - self.stream_hop_len, 'tts_speech_token not equal to {}'.format(self.stream_win_len - self.stream_hop_len)
                yield {'tts_speech': cache_speech}
            p.join()
            torch.cuda.synchronize()
        else:
            # torch.cuda.synchronize()
            # start = time.perf_counter()
            tts_speech_token = []
            for i in self.llm.inference(text=text.to(self.device),
                                                text_len=text_len.to(self.device),
                                                prompt_text=prompt_text.to(self.device),
                                                prompt_text_len=prompt_text_len.to(self.device),
                                                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                prompt_speech_token_len=llm_prompt_speech_token_len.to(self.device),
                                                embedding=llm_embedding.to(self.device).to(self.dtype),
                                                beam_size=1,
                                                sampling=25,
                                                max_token_text_ratio=30,
                                                min_token_text_ratio=3,
                                                stream=stream,
                                                dtype=self.dtype):
                tts_speech_token.append(i)
            assert len(tts_speech_token) == 1, 'tts_speech_token len should be 1 when stream is {}'.format(stream)

            # torch.cuda.synchronize()
            # middle = time.perf_counter()

            tts_speech_token = torch.concat(tts_speech_token, dim=1)
            token_len = torch.tensor([tts_speech_token.size(1)], dtype=torch.int32).to(self.device)
            prompt_token = flow_prompt_speech_token.to(self.device)
            prompt_token_len = flow_prompt_speech_token_len.to(self.device)
            prompt_feat=prompt_speech_feat.to(self.device)
            prompt_feat_len=prompt_speech_feat_len.to(self.device)
            embedding=flow_embedding.to(self.device).to(self.dtype)

            tts_mel = self.flow.inference(token=tts_speech_token,
                                        token_len=token_len,
                                        prompt_token=prompt_token,
                                        prompt_token_len=prompt_token_len,
                                        prompt_feat=prompt_feat,
                                        prompt_feat_len=prompt_feat_len,
                                        embedding=embedding).float()
            tts_speech = self.hift.inference(mel=tts_mel).cpu()
            torch.cuda.empty_cache()

            # torch.cuda.synchronize()
            # end = time.perf_counter()
            # print("time cost: ", middle - start, end - middle)
            yield {'tts_speech': tts_speech}