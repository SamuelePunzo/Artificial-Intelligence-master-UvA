
import unittest
import torch
import torch.nn as nn
from argparse import Namespace
from gpt import GPT, CausalSelfAttention, TransformerDecoderBlock, BERTGELU, RMSNorm

class TestGPTComponents(unittest.TestCase):

    def setUp(self):
        self.config = Namespace()
        self.config.n_embd = 64
        self.config.n_head = 4
        self.config.n_layer = 2
        self.config.vocab_size = 100
        self.config.block_size = 32
        self.config.embd_pdrop = 0.1
        self.config.resid_pdrop = 0.1
        self.config.attn_pdrop = 0.1
        self.config.use_flash_attn = False
        self.config.abs_emb = False # Default to RoPE as per code analysis
        self.config.model_type = 'gpt-nano' # Just for init checks

    def test_bert_gelu(self):
        gelu = BERTGELU()
        x = torch.tensor([-1.0, 0.0, 1.0])
        out = gelu(x)
        # Expected values approx: -0.1587, 0.0, 0.8413
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.allclose(out[1], torch.tensor(0.0)))
        self.assertTrue(out[0] < 0)
        self.assertTrue(out[2] > 0)

    def test_rms_norm(self):
        norm = RMSNorm(self.config.n_embd)
        x = torch.randn(10, self.config.n_embd)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        # Check if variance is close to 1 (RMSNorm normalizes by RMS)
        # RMS = sqrt(mean(x^2)). Output = x / RMS * weight.
        # If weight is 1, RMS(output) should be 1.
        rms = torch.sqrt(torch.mean(out**2, dim=-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))

    def test_causal_self_attention_shape(self):
        attn = CausalSelfAttention(self.config)
        x = torch.randn(2, self.config.block_size, self.config.n_embd) # B, T, C
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_causal_self_attention_masking(self):
        # Ensure future tokens don't affect past tokens
        attn = CausalSelfAttention(self.config)
        attn.eval() # Disable dropout for deterministic check
        
        x = torch.randn(1, 5, self.config.n_embd)
        
        # Forward pass 1
        out1 = attn(x)
        
        # Modify the last token
        x_mod = x.clone()
        x_mod[:, -1, :] = torch.randn(1, self.config.n_embd)
        
        # Forward pass 2
        out2 = attn(x_mod)
        
        # Outputs for the first 4 tokens should be identical
        self.assertTrue(torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-6))
        # Output for the last token should be different
        self.assertFalse(torch.allclose(out1[:, -1, :], out2[:, -1, :]))

    def test_rope_embeddings(self):
        # Test apply_rotary_emb directly
        attn = CausalSelfAttention(self.config)
        B, T, C = 1, 10, self.config.n_embd
        head_size = C // self.config.n_head
        q = torch.randn(B, self.config.n_head, T, head_size)
        k = torch.randn(B, self.config.n_head, T, head_size)
        
        q_rot, k_rot = attn.apply_rotary_emb(q, k, T)
        
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        # Basic check: values should change
        self.assertFalse(torch.allclose(q, q_rot))

    def test_transformer_decoder_block(self):
        block = TransformerDecoderBlock(self.config)
        x = torch.randn(2, self.config.block_size, self.config.n_embd)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_gpt_init(self):
        model = GPT(self.config)
        self.assertIsInstance(model, GPT)
        self.assertEqual(model.transformer.w_token_emb.weight.shape, (self.config.vocab_size, self.config.n_embd))

    def test_gpt_forward(self):
        model = GPT(self.config)
        idx = torch.randint(0, self.config.vocab_size, (2, 10)) # Batch 2, Seq 10
        logits = model(idx)
        self.assertEqual(logits.shape, (2, 10, self.config.vocab_size))

    def test_gpt_generate(self):
        model = GPT(self.config)
        model.eval()
        idx = torch.randint(0, self.config.vocab_size, (1, 5))
        new_tokens = 3
        out = model.generate(idx, max_new_tokens=new_tokens)
        self.assertEqual(out.shape, (1, 5 + new_tokens))

    def test_configure_optimizers(self):
        model = GPT(self.config)
        train_config = Namespace()
        train_config.weight_decay = 0.1
        train_config.learning_rate = 3e-4
        train_config.betas = (0.9, 0.95)
        
        optimizer = model.configure_optimizers(train_config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 2) # Decay and no-decay groups

if __name__ == '__main__':
    unittest.main()
