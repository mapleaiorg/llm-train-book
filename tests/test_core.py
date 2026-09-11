import math
from pathlib import Path
import tempfile
import unittest
import torch
from torch.nn import functional as F
from minillm import Config, TinyLM
from minillm.model import RMSNorm, Attention, rope, cross_entropy
from minillm.data import TokenStream
from minillm.tokenizer import CharacterTokenizer
from minillm.trainer import optimizer_for, train_step, evaluate
from minillm.checkpoint import save, load


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(7)
        self.c = Config(width=16, hidden=24, layers=1, context=8)

    def setup_run(self):
        model = TinyLM(self.c)
        return model, optimizer_for(model), TokenStream(list(range(8)) * 32, 8, 8, 19)

    def test_config_rejects_invalid_shapes(self):
        for args in ({"width": 15}, {"kv_heads": 3}, {"width": 12}, {"epsilon": 0}, {"layers": 0}, {"rope_base": float("nan")}):
            with self.assertRaises(ValueError):
                Config(**args)

    def test_parameter_budget(self):
        c = Config()
        d = c.width // c.query_heads
        expected = 2 * c.vocab_size * c.width + c.layers * (2 * c.width**2 + 2 * c.width * c.kv_heads * d + 3 * c.width * c.hidden + 2 * c.width) + c.width
        self.assertEqual(sum(p.numel() for p in TinyLM(c).parameters()), expected)
        self.assertEqual(expected, 19104)

    def test_book_weighted_sum(self):
        weights = torch.tensor([.1, .55, .25, .1], dtype=torch.double)
        values = torch.tensor([[.2,.4,.1],[.5,.1,.3],[.4,.7,.2],[.1,.3,.8]], dtype=torch.double)
        torch.testing.assert_close(weights @ values, torch.tensor([.405,.3,.305], dtype=torch.double))

    def test_tokenizer_identity_and_roundtrip(self):
        a, b = CharacterTokenizer(), CharacterTokenizer("bacdefgh")
        self.assertEqual(a.decode(a.encode("bad")), "bad")
        self.assertNotEqual(a.identity, b.identity)
        with self.assertRaises(ValueError):
            a.decode([-1])

    def test_shift_and_last_valid_start(self):
        stream = TokenStream([0, 1, 2, 3], 3, 4)
        x, y = stream.batch(2)
        self.assertEqual(x.tolist(), [[0, 1, 2]] * 2)
        self.assertEqual(y.tolist(), [[1, 2, 3]] * 2)
        with self.assertRaises(ValueError):
            TokenStream([0, 1, 2], 3, 4)

    def test_rmsnorm_hand_and_gradient(self):
        norm = RMSNorm(4).double()
        x = torch.tensor([[1., -2., 3., -4.]], dtype=torch.double, requires_grad=True)
        torch.testing.assert_close(norm(x), x / math.sqrt(7.5 + 1e-5))
        self.assertTrue(torch.autograd.gradcheck(norm, (x,)))
        torch.testing.assert_close(norm(torch.zeros_like(x)), torch.zeros_like(x))

    def test_rope_norm_position_and_gradient(self):
        x = torch.randn(2, 2, 3, 4, dtype=torch.double, requires_grad=True)
        result = rope(x)
        torch.testing.assert_close(result.square().sum(-1), x.square().sum(-1))
        torch.testing.assert_close(result[:, :, 0], x[:, :, 0])
        self.assertTrue(torch.autograd.gradcheck(rope, (x,)))

    def test_gqa_independent_head_reference(self):
        attention = Attention(self.c).double()
        x = torch.randn(1, 3, 16, dtype=torch.double)
        q = rope(attention.q(x).view(1, 3, 4, 4).transpose(1, 2))
        k = rope(attention.k(x).view(1, 3, 2, 4).transpose(1, 2))
        v = attention.v(x).view(1, 3, 2, 4).transpose(1, 2)
        outputs = torch.zeros(1, 4, 3, 4, dtype=torch.double)
        for h in range(4):
            for t in range(3):
                scores = torch.stack([torch.dot(q[0, h, t], k[0, h // 2, s]) / 2 for s in range(t + 1)])
                outputs[0, h, t] = sum(p * v[0, h // 2, s] for s, p in enumerate(scores.softmax(0)))
        expected = attention.out(outputs.transpose(1, 2).reshape(1, 3, 16))
        torch.testing.assert_close(attention(x), expected)

    def test_future_tokens_cannot_change_prefix(self):
        model = TinyLM(self.c)
        ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
        altered = ids.clone()
        altered[:, 3:] = torch.tensor([7, 6, 0])
        torch.testing.assert_close(model(ids)[:, :3], model(altered)[:, :3], rtol=0, atol=0)

    def test_invalid_ids(self):
        model = TinyLM(self.c)
        for x in (torch.tensor([[8]]), torch.ones(1, 9, dtype=torch.long), torch.zeros(1, 2)):
            with self.assertRaises(ValueError):
                model(x)

    def test_model_finite_difference(self):
        model = TinyLM(self.c).double()
        x, y = TokenStream(list(range(8)) * 2, 8, 8).batch(1)
        loss = cross_entropy(model(x), y)
        loss.backward()
        parameter = model.head.weight
        analytical = parameter.grad[0, 0].item()
        original = parameter[0, 0].item()
        values = []
        with torch.no_grad():
            for delta in (1e-5, -1e-5):
                parameter[0, 0] = original + delta
                values.append(cross_entropy(model(x), y).item())
            parameter[0, 0] = original
        self.assertAlmostEqual(analytical, (values[0] - values[1]) / 2e-5, places=7)

    def test_unequal_microbatch_accumulation(self):
        model, opt, stream = self.setup_run()
        other, other_opt, _ = self.setup_run()
        other.load_state_dict(model.state_dict())
        x, y = stream.batch(3)
        train_step(model, opt, [(x, y)])
        train_step(other, other_opt, [(x[:1], y[:1]), (x[1:], y[1:])])
        for left, right in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)

    def test_adamw_first_step(self):
        p = torch.nn.Parameter(torch.tensor([2.], dtype=torch.double))
        opt = torch.optim.AdamW([p], lr=.1, weight_decay=.01, eps=1e-8, foreach=False)
        p.grad = torch.tensor([.5], dtype=torch.double)
        opt.step()
        expected = 2 * (1 - .1 * .01) - .1 * .5 / (.5 + 1e-8)
        self.assertAlmostEqual(p.item(), expected, places=12)

    def test_checkpoint_exact_resume_and_identity_rejection(self):
        model, opt, stream = self.setup_run()
        tokenizer = CharacterTokenizer()
        for _ in range(4):
            train_step(model, opt, [stream.batch(2)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step.pt"
            save(path, model, opt, stream, tokenizer, 4)
            expected_losses = [train_step(model, opt, [stream.batch(2)]) for _ in range(4)]
            expected_rng = torch.rand(3)
            restored, restored_opt, restored_stream = self.setup_run()
            self.assertEqual(load(path, restored, restored_opt, restored_stream, tokenizer), 4)
            losses = [train_step(restored, restored_opt, [restored_stream.batch(2)]) for _ in range(4)]
            self.assertEqual(losses, expected_losses)
            self.assertEqual(stream.batches, restored_stream.batches)
            torch.testing.assert_close(torch.rand(3), expected_rng, rtol=0, atol=0)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for index, state in opt.state_dict()["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, restored_opt.state_dict()["state"][index][name], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                load(path, restored, restored_opt, restored_stream, CharacterTokenizer("bacdefgh"))
            with self.assertRaises(ValueError):
                load(path, restored, restored_opt, TokenStream([0] * 30, 8, 8), tokenizer)

    def test_checkpoint_rejects_pending_gradients(self):
        model, opt, stream = self.setup_run()
        x, y = stream.batch(1)
        cross_entropy(model(x), y).backward()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save(Path(directory) / "bad.pt", model, opt, stream, CharacterTokenizer(), 0)

    def test_pattern_learning(self):
        model, opt, stream = self.setup_run()
        heldout = stream.batch(16)
        initial = evaluate(model, *heldout)
        for _ in range(60):
            train_step(model, opt, [stream.batch(8)])
        self.assertLess(evaluate(model, *heldout), initial * .3)

    def test_fresh_random_data_entropy_sanity(self):
        model, opt, _ = self.setup_run()
        train_rng = torch.Generator().manual_seed(100)
        eval_rng = torch.Generator().manual_seed(101)
        for _ in range(100):
            block = torch.randint(8, (16, 9), generator=train_rng)
            train_step(model, opt, [(block[:, :-1], block[:, 1:])])
        block = torch.randint(8, (512, 9), generator=eval_rng)
        value = evaluate(model, block[:, :-1], block[:, 1:])
        self.assertLess(abs(value - math.log(8)), .12)
        logits = torch.zeros(1, 8, 8)
        self.assertAlmostEqual(cross_entropy(logits, torch.arange(8)[None]).item(), math.log(8), places=6)


if __name__ == "__main__":
    unittest.main()
