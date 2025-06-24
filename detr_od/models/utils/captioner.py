#!/usr/bin/env python3
# coding=utf-8
'''
brief        :
Author       : dingbaiyong baiyong.ding@waytous.com
Date         : 2024-06-20 16:36:42
FilePath     : /VLMs_glee/detectron2/projects/glee/models/blip2_encoder/feat_pretrain.py
Description  :
LastEditTime : 2024-07-13 08:59:42
LastEditors  : dingbaiyong
Copyright (c) 2024 by Inc, All Rights Reserved.
'''

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from .blip2base import (
    Blip2Base,
    LayerNorm,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures


class Captioner(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    def __init__(
        self,
        num_query_token=32,
        # num_query_token=196,
        vision_width=1408,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=100,
    ):
        super().__init__()
        self.enc_output = nn.Linear(256, vision_width)
        self.enc_output_norm = LayerNorm(vision_width)
        self.enc_output_2 = nn.Linear(256, 768)
        self.tokenizer = self.init_tokenizer()
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, vision_width, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        self.vision_proj = nn.Linear(
            self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
        self.traing_enable_itm = False
        self.traing_enable_itm = True

    def forward(self, image_embeds, text,  obsquery_tokens=None):
        device = image_embeds.device
        image_embeds = self.enc_output(image_embeds)
        image_embeds = self.enc_output_norm(image_embeds)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.query_tokens.expand(
            image_embeds.shape[0], -1, -1)
        if obsquery_tokens is not None:
            query_tokens = self.enc_output_2(obsquery_tokens)
            query_tokens = self.enc_output_norm(query_tokens)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ### ============== Image-text Contrastive ===================###
        # [batch_size, num_query_tokens, embed_dim] - local batch
        image_feats_local = image_feats
        # [batch_size, num_query_tokens, embed_dim]
        # [batch_size, embed_dim]
        text_feat_local = text_feat
        sim_q2t = torch.matmul(
            image_feats_local.unsqueeze(1), text_feat_local.unsqueeze(-1)
        ).squeeze(-1)
        # [batch_size, batch_size, num_query_tokens]
        # Aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp
        # text-query similarity (only using local batch)
        sim_t2q = torch.matmul(
            text_feat_local.unsqueeze(1).unsqueeze(
                1), image_feats_local.permute(0, 2, 1)
        ).squeeze(-2)

        # Aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size]

        # Set rank and batch size (local batch size)
        # rank = dist.get_rank()
        rank = 0

        bs = image_feats_local.size(0)

        # Define targets for cross-entropy
        targets = torch.linspace(
            rank * bs, rank * bs + bs - 1, bs, dtype=int).to(device)

        # Calculate the contrastive loss
        loss_itc = (
            F.cross_entropy(sim_i2t.clone(), targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i.clone(), targets, label_smoothing=0.1)
        ) / 2
        ### ============== Image-text Matching ===================###
        if self.traing_enable_itm:
            # Use local batch for ITM as well, no need for concat_all_gather here
            text_input_ids_local = text_tokens.input_ids  # local batch
            text_attention_mask_local = text_tokens.attention_mask  # local batch
            image_embeds_local = image_embeds  # local batch

            # Assert no invalid values
            assert torch.isfinite(sim_t2i).all(
            ), "sim_t2i contains invalid values"
            assert torch.isfinite(sim_i2t).all(
            ), "sim_i2t contains invalid values"

            with torch.no_grad():
                # Adjust the similarities for diagonal filling
                sim_t2i.fill_diagonal_(-10000)
                sim_i2t.fill_diagonal_(-10000)

                # Calculate softmax weights for matching
                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_i2t = F.softmax(sim_i2t, dim=1)

                # Ensure no NaN or invalid values in weights
                if torch.any(torch.isnan(weights_t2i)) or torch.any(weights_t2i.sum(dim=1) == 0):
                    raise ValueError(
                        "Invalid weights_t2i for multinomial sampling.")
                if torch.any(torch.isnan(weights_i2t)) or torch.any(weights_i2t.sum(dim=1) == 0):
                    raise ValueError(
                        "Invalid weights_i2t for multinomial sampling.")

            # Select a negative image for each text using weights_t2i
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_local[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # Select a negative text for each image using weights_i2t
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(text_input_ids_local[neg_idx])
                text_atts_neg.append(text_attention_mask_local[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            # Concatenate positive and negative samples
            text_ids_all = torch.cat(
                [text_input_ids_local, text_input_ids_local, text_ids_neg], dim=0)  # pos, pos, neg
            text_atts_all = torch.cat(
                [text_attention_mask_local, text_attention_mask_local, text_atts_neg], dim=0)

            # Query tokens for ITM (local batch)
            query_tokens_itm = self.query_tokens.expand(
                text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[
                                        :-1], dtype=torch.long).to(device)
            attention_mask_all = torch.cat(
                [query_atts_itm, text_atts_all], dim=1)

            # Stack image embeddings (local batch)
            image_embeds_all = torch.cat(
                [image_embeds_local, image_embeds_neg, image_embeds_local], dim=0)  # pos, neg, pos
            image_atts_all = torch.ones(image_embeds_all.size()[
                                        :-1], dtype=torch.long).to(device)

            # Perform ITM head prediction
            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
            )

            if torch.any(torch.isnan(output_itm.last_hidden_state)):
                raise ValueError(
                    "NaN detected in output_itm.last_hidden_state")

            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(
                1), :]
            vl_output = self.itm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0
            ).to(device)

            if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                raise ValueError("Logits contain NaN or Inf")
            if logits.size(0) != itm_labels.size(0):
                raise ValueError(
                    "Mismatch between logits and itm_labels size.")

            loss_itm = F.cross_entropy(logits, itm_labels)

        ## ================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            device
        )
        attention_mask = torch.cat(
            [query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        if not self.traing_enable_itm:
            loss_itm = loss_lm*0
        # loss_itc = loss_lm

        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm), image_feats, query_output.last_hidden_state
        # track_loss = blip_loss.loss_itc
        # dist_loss = blip_loss.loss_itm
        # caption_loss = blip_loss.loss_lm

    @torch.no_grad()
    def generate(
        self,
        image_embeds,
        obsquery_tokens=None,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=100,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        # image_embeds = self.feat_fuser(features)  # (bs, L, C)
        device = image_embeds.device
        # image_atts = torch.ones(
        #     image_embeds.size()[:-1], dtype=torch.long).to(device)
        # # 引入类别文本特征，以增强图像特征，从而学习类别文本特征
        # if text_features is not None:
        #     text_embeds = self.enc_output(text_features["hidden"])
        #     image_embeds = torch.cat(
        #         [image_embeds, text_embeds], dim=1)
        #     image_atts = torch.cat([image_atts, text_features["masks"]], dim=1)

        image_embeds = self.enc_output(image_embeds)
        image_embeds = self.enc_output_norm(image_embeds)
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if obsquery_tokens is not None:
            query_tokens = self.enc_output_2(obsquery_tokens)
            query_tokens = self.enc_output_norm(query_tokens)
            # query_tokens = obsquery_tokens
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return [], None, query_output.last_hidden_state

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )
        if not use_nucleus_sampling:
            hidden_states = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
            hidden_states = image_embeds
        image_atts = torch.ones(hidden_states.size()[:-1], dtype=torch.long).to(
            hidden_states.device
        )
        model_kwargs = {
            "encoder_hidden_states": hidden_states,
            "encoder_attention_mask": image_atts,
        }
        input_ids = (
            torch.LongTensor(image_embeds.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(hidden_states.device)
        )
        query_tokens = self.query_tokens.expand(
            hidden_states.shape[0], -1, -1)
        if obsquery_tokens is not None:
            query_tokens = self.enc_output_2(obsquery_tokens)
            query_tokens = self.enc_output_norm(query_tokens)
            # query_tokens = obsquery_tokens
            query_tokens = query_tokens.repeat_interleave(num_beams, dim=0)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        return captions, image_feats, query_output.last_hidden_state

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(
            1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, image_feat, caption, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image_feat is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.input_image_proj(
                    image_feat).flatten(-2).permute(0, 2, 1))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(
                self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.input_image_proj(
                    image_feat).flatten(-2).permute(0, 2, 1))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat(
                [query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(
                1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
