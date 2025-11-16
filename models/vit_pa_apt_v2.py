'''
 * Updated for PA-APT V2: Layer-wise Patch Selection
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv

def printdata(x, name, layer,flag=False):
    if flag:
        std_x = torch.std(x)
        print(f'{name},std,{layer},{std_x.item():.6f}')
        mean_x = torch.mean(x)
        print(f'{name},mean,{layer},{mean_x.item():.6f}')
       
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        
        self.k = 5 
        
        # New learnable prompts for the top-k patches
        self.patch_p_k = nn.Parameter(torch.zeros(1, num_heads, 1, head_dim))
        self.patch_p_v = nn.Parameter(torch.zeros(1, num_heads, 1, head_dim))
        
        trunc_normal_(self.patch_p_k, std=.02)
        trunc_normal_(self.patch_p_v, std=.02)
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    # MODIFIED: Accepts k_patches and patch_selector for V2 logic
    def forward(self, x, register_hook=False, prompt=None, layer=-1, k_patches=5, patch_selector=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # Shape: [B, H, N, D_h]
        
        # --- PA-APT V2 LOGIC START ---
        
        # 1. Determine k for this layer using the selector method
        if patch_selector is not None:
            # We call the select_patches_per_layer method from the learner object
            current_k = patch_selector(layer)
        else:
            current_k = k_patches
            
        k_mod = k.clone()
        v_mod = v.clone()

        # 2. PROBE PASS (Only run if patches are needed, current_k > 0)
        if current_k > 0:
            attn_scores_probe = (q @ k.transpose(-2, -1)) * self.scale
            attn_softmax_probe = attn_scores_probe.softmax(dim=-1) 
            
            # 3. SELECT TOP-K
            cls_to_patch_scores = attn_softmax_probe.mean(dim=1)[:, 0, 1:] 
            
            # Get the indices of the top-k patches (using current_k)
            topk_indices = torch.topk(cls_to_patch_scores, k=current_k, dim=-1)[1]
            
            # Add 1 to indices to account for skipping the CLS token
            topk_indices = topk_indices + 1 
        
        # 4. APPLY PROMPTS
        if prompt is not None:
            if type(prompt) is list and len(prompt) == 2:
                pk, pv = prompt # These are the original CLS prompts
                
                # 4a. Apply original CLS prompts (index 0)
                k_mod[:,:,0:1] = k_mod[:,:,0:1] + pk[:,:,0:1]
                v_mod[:,:,0:1] = v_mod[:,:,0:1] + pv[:,:,0:1]
                
                # 4b. Apply new Patch Prompts using scatter_add_ (Only if current_k > 0)
                if current_k > 0:
                    B_h, H, N_k, D_h = k.shape
                    
                    # Expand indices to match K/V shape for scatter_add
                    idx_expanded = topk_indices.unsqueeze(1).unsqueeze(-1).expand(B_h, H, current_k, D_h)
                    
                    # Expand patch prompts to be the source
                    patch_pk_src = self.patch_p_k.expand(B_h, H, current_k, D_h)
                    patch_pv_src = self.patch_p_v.expand(B_h, H, current_k, D_h)

                    # Add prompts to K and V at the top-k indices
                    k_mod.scatter_add_(dim=2, index=idx_expanded, src=patch_pk_src)
                    v_mod.scatter_add_(dim=2, index=idx_expanded, src=patch_pv_src)
                
            else:
                raise ValueError("prompt type not supported!")
        
        # 5. FINAL PASS: Re-compute attention with *modified* K and V
        attn = (q @ k_mod.transpose(-2, -1)) * self.scale
        # --- END PA-APT MODIFICATION ---
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v_mod).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    # MODIFIED V2: Pass k_patches and patch_selector down
    def forward(self, x, register_hook=False, prompt=None, layer=-1, k_patches=5, patch_selector=None):
        if prompt is not None:
            if type(prompt) is list: # prompt is [P_k, P_v, learner_instance]
                # MODIFIED: Pass V2 params to Attention layer
                _x, attn = self.attn(
                    self.norm1(x), 
                    register_hook=register_hook, 
                    prompt=prompt[:2], # <-- FIX: Pass only [P_k, P_v]
                    layer=layer,
                    k_patches=k_patches, 
                    patch_selector=patch_selector
                )
                x = x + self.drop_path(_x)
            else:
                raise ValueError("prompt type not suported!")
        else:
            _x, attn = self.attn(self.norm1(x), register_hook=register_hook, prompt=None,layer=layer)
            x = x + self.drop_path(_x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn
    
class VisionTransformer(nn.Module):
    
    # MODIFIED V2: Accept k_patches and use_layer_wise from learner
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 ckpt_layer=0, k_patches=5, use_layer_wise=False): # V2 Params added here
        
        super().__init__()
        self.num_features = self.embed_dim = embed_dim 
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Store V2 settings
        self.k_patches = k_patches 
        self.use_layer_wise = use_layer_wise
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # MODIFIED V2: The prompt object carries the selector logic
    def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, task_id=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        prepend_layers =[]
        add_layers = [0,1,2,3,4,5,6,7,8,9,10,11]

        if prompt is None:
            for i, blk in enumerate(self.blocks):
                x, attn = blk(x, register_blk==i)
        else:
        # V2: The prompt object (APT) is passed
            prompt_obj = prompt 

            for i, blk in enumerate(self.blocks):
                if i in prepend_layers:
                    pass
                elif i in add_layers:                            
                    # prompt_list is now [P_k, P_v, learner_instance]
                    prompt_list = prompt_obj.forward(i, x, train=train)

                # MODIFIED: Extract the patch_selector (learner_instance) from the list
                patch_selector = prompt_list[2] 

                x, attn = blk(
                    x, 
                    register_blk==i, 
                    prompt=prompt_list, # Pass the full list
                    layer=i,
                    k_patches=self.k_patches, 
                    patch_selector=patch_selector # Pass the correct selector
                )

        x = self.norm(x)

        return x

    @torch.no_grad()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
        

@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))

            
def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    
    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint
