"""Implementation of the Neural Interpreter.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from torchinfo import summary


class TypeInference(nn.Module):
    """Compute the type vector.
    The vector is normalized.
    """
    def __init__(self, d_emb: int, n_layers: int, d_signature: int):
        super().__init__()

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_emb, d_emb),
                nn.GELU(),
            )
            for _ in range(n_layers)
        ])

        self.last_linear = nn.Linear(d_emb, d_signature)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        x = self.last_linear(x)
        return F.normalize(x, p=2, dim=-1)


class TypeMatching(nn.Module):
    def __init__(self, d_emb: int, n_layers: int, d_signature: int):
        super().__init__()

        self.type_inference = TypeInference(d_emb, n_layers, d_signature)
        self.sigma = nn.Parameter(torch.randn(1))

    def forward(self, x: torch.Tensor, s: torch.Tensor, threshold: float):
        """Compute the compatibility matrix C.

        Args
        ----
            x: Token embeddings.
                Size of [batch_size, n_token, embedding_size]
            s: Function signatures.
                Size of [n_function, signature_size]
            threshold: Threshold for the compatibility score.

        Output
        ------
            C: Compatibility matrix. C[u, i] is the compatibility
               between function f_u and token x_i.
                Size of [batch_size, n_token, n_function]
        """
        t = self.type_inference(x)  # [batch_size, n_token, signature_size]
        s = repeat(s, 'f s -> b f s', b=t.shape[0])  # [batch_size, n_function, signature_size]
        d = 1 - torch.einsum('bts, bfs -> btf', t, s)  # [batch_size, n_token, n_function]

        mask = d < threshold  # [Batch_size, n_token, n_function]
        d = -d / self.sigma
        d = d.masked_fill(~mask, float('-inf'))  # Replace values where mask is False to -oo

        return torch.softmax(d, dim=-1)


class ModLin(nn.Module):
    """Linear layer modulated by a conditional vector.
    It is actually a stack of the original ModLin layer. There is
    one ModLin for each function.

    TODO: How are the parameters shared accross the functions?
    """
    def __init__(self, d_emb: int, d_cond: int, n_function: int):
        super().__init__()
        self.n_function = n_function

        self.linear = nn.Linear(d_emb * n_function, d_emb * n_function)
        self.cond = nn.Sequential(
            nn.Linear(d_cond * n_function, d_emb * n_function, bias=False),
            nn.LayerNorm(d_emb * n_function)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """Pass the tokens in each ModLin functions conditioned by c.

        Args
        ----
            x: Token embeddings for each function.
                Shape of [batch_size, n_token, n_function, embedding_size]
            c: Function codes.
                Shape of [n_function, code_embedding]

        Output
        ------
            y: Output of all functions conditioned by c.
                Shape of [batch_size, n_token, n_function, embedding_size]
        """
        c = repeat(c, 'f c -> b f c', b=x.shape[0])  # [batch_size, n_function, code_embedding]
        c = rearrange(c, 'b f c -> b (f c)')  # [batch_size, n_function * code_embedding]
        c = self.cond(c)  # [batch_size, n_function * embedding_size]
        c = repeat(c, 'b e -> b t e', t=x.shape[1])  # [batch_size, n_token, n_function * embedding_size]

        x = rearrange(x, 'b t f e -> b t (f e)')
        x = self.linear(x * c)

        x = rearrange(x, 'b t (f e) -> b t f e', f=self.n_function)
        return x

    def repeat_x(self, x: torch.Tensor):
        """Prepare the input x to be fed to this ModLin module.

        Args
        ----
            x: Token embeddings.
                Shape of [batch_size, n_token, embedding_size]

        Output
        ------
            x: Token embeddings of each function.
                Shape of [batch_size, n_token, n_function, embedding_size]
        """
        return repeat(x, 'b t e -> b t f e', f=self.n_function)


class ModMLP(nn.Module):
    """MLP where linear layers are replaced by ModLin layers.
    """
    def __init__(self, d_emb: int, d_cond: int, n_function: int, n_layers: int):
        super().__init__()

        assert n_layers >= 1

        self.mlp = nn.ModuleList([
            nn.ModuleList([
                ModLin(d_emb, d_cond, n_function),
                nn.GELU(),
            ])
            for _ in range(n_layers-1)
        ])

        self.last_linear = ModLin(d_emb, d_cond, n_function)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """Chains multiple ModLin layers across the multiple functions.

        Args
        ----
            x: Token embeddings for each function.
                Shape of [batch_size, n_token, n_function, embedding_size]
            c: Function codes.
                Shape of [n_function, code_embedding]

        Output
            y: Output of token embeddings for each function.
                Shape of [batch_size, n_token, n_function, embedding_size]
        """
        for linear, activation in self.mlp:
            x = linear(x, c)
            x = activation(x)
        return self.last_linear(x, c)

    def repeat_x(self, x: torch.Tensor):
        return self.last_layer.repeat_x(x)


class ModAttn(nn.Module):
    """Attention module where linear layers are replaced by ModLin layers.

    TODO: implement it in a multihead fashion.
    """
    def __init__(self, d_emb: int, d_cond: int, n_function: int):
        super().__init__()

        self.Q = ModLin(d_emb, d_cond, n_function)
        self.K = ModLin(d_emb, d_cond, n_function)
        self.V = ModLin(d_emb, d_cond, n_function)

        self.linear = ModLin(d_emb, d_cond, n_function)

    def forward(self, x: torch.Tensor, c: torch.Tensor, C: torch.Tensor):
        """Compute the modulated attention between the tokens.

        Args
        ----
            x: Token embeddings for each function.
                Shape of [batch_size, n_token, n_function, embedding_size]
            c: Function codes.
                Shape of [n_function, code_size]
            C: Compatibility matrix.
                Shape of [batch_size, n_token, n_function]

        Output
        ------
            y: Token embeddings for each function.
                Shape of [batch_size, n_token, n_function, embedding_size]
        """
        q = self.Q(x, c)  # [batch_size, n_token, n_function, embedding_size]
        k = self.K(x, c)
        v = self.V(x, c)

        attention = torch.einsum('bqfe, bkfe -> bfqk', q, k)  # [batch_size, n_function, n_token, n_token]
        attention = torch.softmax(attention / np.sqrt(k.shape[-1]), dim=-1)

        # To compute P, we do the outer product of C_u
        # Where C_u is [C_0u, ..., C_Tu] the compatibility of each token for the given function u
        # P is then the matrix of each pair of product C_ui * C_uj
        P = torch.einsum('btf, bqf -> bftq', C, C)  # [batch_size, n_function, n_token, n_token]
        W = torch.softmax(P * attention, dim=-1)

        y = torch.einsum('bftq, bufe -> btfe', W, v)  # [batch_size, n_token, n_function, embedding_size]
        y = self.linear(y, c)
        return y

    def repeat_x(self, x: torch.Tensor):
        return self.linear.repeat_x(x)


class LineOfCode(nn.Module):
    """ModAttn layer followed by a ModMLP layer.
    """
    def __init__(
            self,
            d_emb: int,
            d_cond: int,
            n_function: int,
            n_layers: int,
        ):
        super().__init__()
        self.d_emb = d_emb

        self.attention = ModAttn(d_emb, d_cond, n_function)
        self.mlp = ModMLP(d_emb, d_cond, n_function, n_layers)
        self.norm_x = nn.LayerNorm(d_emb)
        self.norm_a = nn.LayerNorm(d_emb)

    def forward(self, x: torch.Tensor, c: torch.Tensor, C: torch.Tensor):
        """Compute a line of code, which is a modulated attention followed by a modulated MLP layer.
        Everything is pondered by the type matching mecanisme.

        Args
        ----
            x: Token embeddings for each function.
                Shape of [batch_size, n_token, n_function, embedding_size]
            c: Function codes.
                Shape of [n_function, code_size]
            C: Compatibility matrix.
                Shape of [batch_size, n_token, n_function]

        Output
        ------
            b: Token embeddings for each function.
                Shape of [batch_size, n_token, n_function, embedding_size]
        """
        mod = repeat(C, 'b t f -> b t f e', e=self.d_emb)
        a = self.attention(self.norm_x(x), c, C)  # [batch_size, n_token, n_function, embedding_size]
        a = x + mod * a

        b = self.mlp(self.norm_a(a), c)  # [batch_size, n_token, n_function, embedding_size]
        b = a + mod * b
        return b

    def repeat_x(self, x: torch.Tensor):
        return self.attention.repeat_x(x)


class Interpreter(nn.Module):
    """Stack of multiple lines of code, sharing the same codes function.
    """
    def __init__(
        self, 
        d_emb: int,
        d_cond: int,
        n_function: int,
        n_layers_mlp: int,
        n_lines: int,
    ):
        super().__init__()

        self.loc_stack = nn.ModuleList([
            LineOfCode(d_emb, d_cond, n_function, n_layers_mlp)
            for _ in range(n_lines)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor, C: torch.Tensor):
        """Pass the tokens into all the line of code.
        Everything is pondered by the type matching mecanisme.

        Args
        ----
            x: Token embeddings.
                Shape of [batch_size, n_token, embedding_size]
            c: Function codes.
                Shape of [n_function, code_size]
            C: Compatibility matrix.
                Shape of [batch_size, n_token, n_function]

        Output
        ------
            y: Token embeddings after all line of codes.
                Shape of [batch_size, n_token, embedding_size]
        """
        v = self.loc_stack[0].repeat_x(x)  # [batch_size, n_token, n_function, embedding_size]
        for loc in self.loc_stack:
            v = loc(v, c, C)

        v = torch.einsum('btfe, btf -> bte', v, C)  # [batch_size, n_token, embedding_size]
        y = x + v
        return y


class FunctionIteration(nn.Module):
    """Combine the interpreter with the type matching mecanism.
    Create the code and signature function.
    """
    def __init__(
        self, 
        d_emb: int,
        d_cond: int,
        n_function: int,
        n_layers_mlp: int,
        n_lines: int,
        d_signature: int,
    ):
        super().__init__()

        self.codes = nn.Parameter(
            torch.randn((n_function, d_cond))
        )
        self.signatures = nn.Parameter(
                torch.randn((n_function, d_signature))
        )

        self.interpreter = Interpreter(d_emb, d_cond, n_function, n_layers_mlp, n_lines)
        self.type_matching = TypeMatching(d_emb, n_layers_mlp, d_signature)

    def forward(self, x: torch.Tensor, threshold: float):
        """Compute the type of each tokens and pass them into an interpreter.

        Args
        ----
            x: Token embeddings.
                Shape of [batch_size, n_token, embedding_size]
            threshold: Threshold for the type type matching mechanism.

        Output
        ------
            y: Token embeddings after the interpreter.
                Shape of [batch_size, n_token, embedding_size]
        """
        C = self.type_matching(x, self.signatures, threshold)
        y = self.interpreter(x, self.codes, C)
        return y


class Script(nn.Module):
    """Stack multiple function iterations.
    """
    def __init__(
        self, 
        d_emb: int,
        d_cond: int,
        n_function: int,
        n_layers_mlp: int,
        n_lines: int,
        d_signature: int,
        n_fn_iteration: int,
    ):
        super().__init__()

        self.fn_iterations = nn.ModuleList([
            FunctionIteration(d_emb, d_cond, n_function, n_layers_mlp, n_lines, d_signature)
            for _ in range(n_fn_iteration)
        ])

    def forward(self, x: torch.Tensor, threshold: float):
        """Pass the tokens through the multiple function iterations.

        Args
        ----
            x: Token embeddings.
                Shape of [batch_size, n_token, embedding_size]
            threshold: Threshold for the type matching mechanism.

        Output
        ------
            y: Token embeddings after the multiple function iterations.
                Shape of [batch_size, n_token, embedding_size]
        """
        for fn_iteration in self.fn_iterations:
            x = fn_iteration(x, threshold)
        return x


if __name__ == '__main__':
    config = {
        'n_function': 5,
        'n_layers_mlp': 3,
        'n_lines': 2,
        'n_fn_iteration': 3,
        'd_emb': 50,
        'd_cond': 30,
        'd_signature': 30,
    }
    other = {
        'b_size': 128,
        'n_token': 10,
        'threshold': 0.1
    }

    model = Script(**config)
    x = torch.randn((other['b_size'], other['n_token'], config['d_emb']))
    summary(model, input_data=[x, other['threshold']])
