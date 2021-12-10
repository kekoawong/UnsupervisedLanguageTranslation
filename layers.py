"""Some neural network layers that you can use in building your translation system.

Many of these layers are tailored for low-resource translation (thanks to Toan Nguyen)."""

import torch

def bmv(w, x):
    """Matrix-vector multiplication that works even if x is a sequence of
    vectors.

    If w has size m,n and x has size n, performs a standard
    matrix-vector multiply, yielding a vector of size m.

    If w has size m,n and x has size b,n, multiplies w with every
    column of x, yielding a matrix of size b,m.
    """
    
    x = x.unsqueeze(-1)
    y = w @ x
    y = y.squeeze(-1)
    return y

class Embedding(torch.nn.Module):
    """Embedding layer.

    The constructor takes arguments:
        vocab_size: Vocabulary size (int)
        output_dims: Size of output vectors (int)

    The resulting Embedding object is callable. See the documentation
    for forward().
    """

    def __init__(self, vocab_size, output_dims):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(vocab_size, output_dims))
        torch.nn.init.normal_(self.W, std=0.01)

    def forward(self, inp):
        """Works on either single words or sequences of words.

        Argument:
            inp: Word (int in {0,...,vocab_size-1})

        Return:
            Word embedding (tensor of size output_dims)

        *or*

        Argument:
            inp: Words (tensor of size n, elements are ints in {0,...,vocab_size-1})

        Return:
            Word embeddings (tensor of size n,output_dims)
        """

        if not (isinstance(inp, int) or inp.dtype in [torch.int32, torch.int64]):
            raise TypeError('input should be an integer or tensor of integers')
        
        emb = self.W[inp]
        
        # Scaling the embedding to have norm 1 helps against overfitting.
        # https://www.aclweb.org/anthology/N18-1031/
        emb = torch.nn.functional.normalize(emb, dim=-1)
        
        return emb

class RNN(torch.nn.Module):
    """Simple recurrent neural network.

    The constructor takes one argument:
        dims: Size of both the input and output vectors (int)

    The resulting RNN object can be used in two ways:
      - Step by step, using start(), input(), and output()
      - On a whole sequence at once, using sequence()
    Please see the documentation for those methods.

    This implementation adds a _residual connection_, which just means
    that output vector is the standard output vector plus the input
    vector. This helps against overfitting, but makes the
    implementation slightly more complicated.
    """
    
    def __init__(self, dims):
        super().__init__()
        self.h0 = torch.nn.Parameter(torch.empty(dims))
        self.W_hi = torch.nn.Parameter(torch.empty(dims, dims))
        self.W_hh = torch.nn.Parameter(torch.empty(dims, dims))
        self.b = torch.nn.Parameter(torch.empty(dims))
        torch.nn.init.normal_(self.h0, std=0.01)
        torch.nn.init.normal_(self.W_hi, std=0.01)
        torch.nn.init.normal_(self.W_hh, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)

    def start(self):
        """Return the initial state, which is a pair consisting of the
        most recent input (here 0 because there's no previous input) and h."""
        return (0, self.h0)

    def input(self, state, inp):
        """Given the old state (state), read in an input vector (inp) and
        compute the new state.

        Arguments:
            state:  Pair consisting of h and input (tensor of size dims)
            inp:    Input vector (tensor of size dims)

        Returns:    Pair consisting of new h and inp

        """

        _, h_prev = state
        
        dims = self.b.size()[0]
        if h_prev.size()[-1] != dims:
            raise TypeError(f'Previous hidden-state vector must have size {dims}')
        if inp.size()[-1] != dims:
            raise TypeError(f'Input vector must have size {dims}')
        
        h_cur = torch.tanh(bmv(self.W_hi, inp) + bmv(self.W_hh, h_prev) + self.b)
        return (inp, h_cur)

    def output(self, state):
        """Compute an output vector based on the state.

        Arguments:
            state:  Pair consisting of h and input (tensor of size dims)
        Returns a pair (out, h_cur), where:
            out:    Output vector (tensor of size dims)
        """
        
        inp, h = state
        return inp + h

    def sequence(self, inputs):
        """Run the RNN on an input sequence.

        Argument:
            Input vectors (tensor of size n,dims)

        Return:
            Output vectors (tensor of size n,dims)
        """
        dims = self.b.size()[0]
        if inputs.ndim != 2:
            raise TypeError("inputs must have exactly two axes")
        if inputs.size()[1] != dims:
            raise TypeError(f'Input vectors must have size {dims}')

        h = self.start()
        outputs = []
        for inp in inputs:
            h = self.input(h, inp)
            o = self.output(h)
            outputs.append(o)
        return torch.stack(outputs)
    
class TanhLayer(torch.nn.Module):
    """Tanh layer.

    The constructor takes these arguments:
        input_dims:  Size of input vectors (int)
        output_dims: Size of output vectors (int)
        residual:    Add a residual connection (bool)

    The resulting TanhLayer object is callable. See forward().

    If residual is True, then input_dims and output_dims must be equal.
    """
    def __init__(self, input_dims, output_dims, residual=False):
        super().__init__()
        self.residual = residual
        if residual and input_dims != output_dims:
            raise ValueError("A residual connection requires the same number of input and output dimensions.")
        self.W = torch.nn.Parameter(torch.empty(output_dims, input_dims))
        self.b = torch.nn.Parameter(torch.empty(output_dims))
        torch.nn.init.normal_(self.W, std=0.01)
        torch.nn.init.normal_(self.b, std=0.01)

    def forward(self, inp):
        """Works on either single vectors or sequences of vectors.

        Argument:
            inp: Input vector (tensor of size input_dims)

        Return:
            Output vector (tensor of size output_dims)

        *or*

        Argument:
            inp: Input vectors (tensor of size n,input_dims)

        Return:
            Output vectors (tensor of size n,output_dims)
        """
        
        input_dims = self.W.size()[-1]
        if inp.size()[-1] != input_dims:
            raise TypeError(f"The inputs must have size {input_dims} (not {inp.size()[-1]})")
        
        out = torch.tanh(bmv(self.W, inp) + self.b)
        if self.residual:
            out = out + inp
        return out

class SoftmaxLayer(torch.nn.Module):
    """Softmax layer.

    The constructor takes these arguments:
        input_dims:  Size of input vectors (int)
        output_dims: Size of output vectors (int)

    The resulting SoftmaxLayer is callable (see forward()).
    """
    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(output_dims, input_dims))
        torch.nn.init.normal_(self.W, std=0.01)

    def forward(self, inp):
        """Works on either single vectors or sequences of vectors.

        Argument:
            inp: Input vector (tensor of size input_dims)

        Return:
            Vector of log-probabilities (tensor of size output_dims)

        *or*

        Argument:
            inp: Input vectors (tensor of size n,input_dims)

        Return:
            Vectors of log-probabilities (tensor of size n,output_dims)
        """

        input_dims = self.W.size()[-1]
        if inp.size()[-1] != input_dims:
            raise TypeError(f"The inputs must have size {input_dims}")
        
        # Scaling both the output embeddings and the inputs
        # to have norm 1 and 10, respectively, helps against overfitting.
        # https://www.aclweb.org/anthology/N18-1031/
        W = torch.nn.functional.normalize(self.W, dim=1)
        inp = torch.nn.functional.normalize(inp, dim=-1) * 10

        return torch.log_softmax(bmv(W, inp), dim=-1)

def attention(query, keys, vals):
    """Compute dot-product attention.

    query can be a single vector or a sequence of vectors.

    Arguments:
        keys:  Key vectors (tensor with size n,d)
        query: Query vector (tensor with size d)
        vals:  Value vectors (tensor with size n,d')

    Returns:
        Context vector (tensor with size d')

    *or*

    Arguments:
        keys:  Key vectors (tensor with size n,d)
        query: Query vectors (tensor with size m,d)
        vals:  Value vectors (tensor with size n,d')

    Returns:
        Context vectors (tensor with size m,d')
    """
    
    if query.size()[-1] != keys.size()[-1]:
        raise TypeError("The queries and keys should be the same size (last axis)")
    if keys.size()[-2] != vals.size()[-2]:
        raise TypeError("There must be the same number of keys and values (second-to-last axis)")

    logits = query @ keys.transpose(-2, -1)  # m,n
    aweights = torch.softmax(logits, dim=-1) # m,n
    context = aweights @ vals                # m,d
    return context

class SelfAttention(torch.nn.Module):
    """Self-attention layer, for use in an encoder.

    The SelfAttention constructor takes one argument:
        dims: Size of input and output vectors (int)

    The resulting object is callable (see forward()) but can only be
    used on sequences of vectors, not single vectors.
    """
    
    def __init__(self, dims):
        super().__init__()
        self.W_Q = torch.nn.Parameter(torch.empty(dims, dims))
        self.W_K = torch.nn.Parameter(torch.empty(dims, dims))
        self.W_V = torch.nn.Parameter(torch.empty(dims, dims))
        torch.nn.init.normal_(self.W_Q, std=0.01)
        torch.nn.init.normal_(self.W_K, std=0.01)
        torch.nn.init.normal_(self.W_V, std=0.01)

    def forward(self, inputs):
        """Argument:
            inputs: Input vectors (tensor of size n,d)

        Return:
            Output vectors (tensor of size n,d)
        """

        dims = self.W_Q.size()[0]
        if inputs.ndim < 2:
            raise TypeError("inputs must have at least two axes")
        if inputs.size()[-1] != dims:
            raise TypeError(f"input vectors must have size {dims}")

        # Linearly transform inputs in three ways to get queries, keys, values
        queries = bmv(self.W_Q, inputs)
        keys = bmv(self.W_K, inputs)
        values = bmv(self.W_V, inputs)

        # Compute output vectors
        outputs = attention(queries, keys, values)
        
        # Residual connection (see RNN for explanation)
        outputs = outputs + inputs
        
        return outputs

class MaskedSelfAttention(torch.nn.Module):
    """Masked self-attention layer, for use in a decoder.

    The MaskedSelfAttention constructor takes one argument:
        dims: Size of input and output vectors (int)

    The resulting object has start(), input(), and output() methods;
    please see documentation for those methods.
    """
    
    def __init__(self, dims):
        super().__init__()
        self.W_Q = torch.nn.Parameter(torch.empty(dims, dims))
        self.W_K = torch.nn.Parameter(torch.empty(dims, dims))
        self.W_V = torch.nn.Parameter(torch.empty(dims, dims))
        self.initial = torch.nn.Parameter(torch.empty(dims))
        torch.nn.init.normal_(self.W_Q, std=0.01)
        torch.nn.init.normal_(self.W_K, std=0.01)
        torch.nn.init.normal_(self.W_V, std=0.01)
        torch.nn.init.normal_(self.initial, std=0.01)

    def start(self):
        """Return the initial list of previous inputs.

        For MaskedSelfAttention, the "state" is the list of previous
        inputs.
        """

        # Initially there are no previous inputs, but we have to have
        # something. Most implementations avoid this problem by
        # prepending <BOS>; here, the parameter self.initial is
        # basically the encoding of <BOS>.
        dims = self.initial.size()[0]
        return torch.empty(0, dims)

    def input(self, prev_inps, inp):
        """Read in an input vector and append it to the list of previous inputs."""
        inputs = torch.cat([prev_inps, inp.unsqueeze(0)], dim=0)

        return inputs
        
    def output(self, inputs):
        """Compute an output vector based on the new list of inputs."""

        # Linearly transform inputs in three ways to get queries, keys, values
        query = bmv(self.W_Q, inputs[-1])
        keys = bmv(self.W_K, inputs)
        values = bmv(self.W_V, inputs)

        # Compute output vectors
        output = attention(query, keys, values)
        
        # Residual connection (see RNN for explanation)
        output = output + inputs[-1]

        return output
