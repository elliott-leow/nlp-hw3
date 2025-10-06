#!/usr/bin/env python3
# CS465 at Johns Hopkins University.
# Module to estimate n-gram probabilities.

# Updated by Jason Baldridge <jbaldrid@mail.utexas.edu> for use in NLP
# course at UT Austin. (9/9/2008)

# Modified by Mozhi Zhang <mzhang29@jhu.edu> to add the new log linear model
# with word embeddings.  (2/17/2016)

# Refactored by Arya McCarthy <xkcd@jhu.edu> because inheritance is cool
# and so is separating business logic from other stuff.  (9/19/2019)

# Patched by Arya McCarthy <arya@jhu.edu> to fix a counting issue that
# evidently was known pre-2016 but then stopped being handled?

# Further refactoring by Jason Eisner <jason@cs.jhu.edu> 
# and Brian Lu <zlu39@jhu.edu>.  (9/26/2021)

from __future__ import annotations

import logging
import math
import pickle
import sys

from pathlib import Path

import torch
from torch import nn
from torch import optim
from jaxtyping import Float
from typeguard import typechecked
from typing import Counter, Collection
from collections import Counter

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab    = Collection[Wordtype]   # and change this to Integerizer[str]
Zerogram = Tuple[()]
Unigram  = Tuple[Wordtype]
Bigram   = Tuple[Wordtype, Wordtype]
Trigram  = Tuple[Wordtype, Wordtype, Wordtype]
Ngram    = Union[Zerogram, Unigram, Bigram, Trigram]
Vector   = List[float]
TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    # OPTIONAL SPEEDUP: You may want to modify this to integerize the
    # tokens, using integerizer.py as in previous homeworks.
    # In that case, redefine `Wordtype` from `str` to `int`.

    # PYTHON NOTE: This function uses `yield` to return the tokens one at
    # a time, rather than constructing the whole sequence and using
    # `return` to return it.
    #
    # A function that uses `yield` is called a "generator."  As with other
    # iterators, it computes new values only as needed.  The sequence is
    # never fully constructed as an single object in memory.
    #
    # You can iterate over the yielded sequence, for example, like this:
    #      for token in read_tokens(my_file, vocab):
    #          process(token)
    # Whenever the `for` loop needs another token, read_tokens magically picks up 
    # where it left off and continues running until the next `yield` statement.

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def draw_trigrams_forever(file: Path, 
                          vocab: Vocab, 
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for 
    SGD training.  
    
    If randomize is True, then randomize the order of the trigrams each time.  
    This is more in the spirit of SGD, but the randomness makes the code harder to debug, 
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random
        pool = tuple(trigrams)   
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram

##### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    """Read vocabulary from either a simple vocab file or an embedding file.
    
    Embedding files start with a header line like '75 10' (vocab_size dim).
    Simple vocab files have one word per line.
    """
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        first_line = f.readline().strip()
        
        # Check if it's an embedding file (header format: "NUM NUM")
        parts = first_line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            # It's an embedding file - read words from first column of remaining lines
            for line in f:
                line = line.strip()
                if line:
                    word = line.split('\t')[0]  # First column is the word
                    vocab.add(word)
        else:
            # It's a simple vocab file - first line is already a word
            vocab.add(first_line)
            for line in f:
                word = line.strip()
                if word:
                    vocab.add(word)
    
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    # Convert from an unordered Set to an ordered List.  This ensures that iterating
    # over the vocab will always hit the words in the same order, so that you can 
    # safely store a list or tensor of embeddings in that order, for example.
    return sorted(vocab)   
    # Alternatively, you could choose to represent a Vocab as an Integerizer (see above).
    # Then you won't need to sort, since Integerizers already have a stable iteration order.

##### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0   # To print progress.

        self.event_count:   Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram, 
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]  
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    # We need to collect two kinds of n-gram counts.
    # To compute p(z | xy) for a trigram xyz, we need c(xy) for the 
    # denominator and c(yz) for the backed-off numerator.  Both of these 
    # look like bigram counts ... but they are not quite the same thing!
    #
    # For a sentence of length N, we are iterating over trigrams xyz where
    # the position of z falls in 1 ... N+1 (so z can be EOS but not BOS),
    # and therefore
    # the position of y falls in 0 ... N   (so y can be BOS but not EOS).
    # 
    # When we write c(yz), we are counting *events z* with *context* y:
    #         c(yz) = |{i in [1, N]: w[i-1] w[i] = yz}|
    # We keep these "event counts" in `event_count` and use them in the numerator.
    # Notice that z=BOS is not possible (BOS is not a possible event).
    # 
    # When we write c(xy), we are counting *all events* with *context* xy:
    #         c(xy) = |{i in [1, N]: w[i-2] w[i-1] = xy}|
    # We keep these "context counts" in `context_count` and use them in the denominator.
    # Notice that y=EOS is not possible (EOS cannot appear in the context).
    #
    # In short, c(xy) and c(yz) count the training bigrams slightly differently.  
    # Likewise, c(y) and c(z) count the training unigrams slightly differently.
    #
    # Note: For bigrams and unigrams that don't include BOS or EOS -- which
    # is most of them! -- `event_count` and `context_count` will give the
    # same value.  So you could save about half the memory if you were
    # careful to store those cases only once.  (How?)  That would make the
    # code slightly more complicated, but would be worth it in a real system.

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z )] += 1
        self.event_count[   (y, z )] += 1
        self.event_count[      (z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[        ()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion, 
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram    # we don't care about z
        self.context_count[(x, y )] += 1
        self.context_count[   (y,)] += 1
        self.context_count[     ()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly, 
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )

    def save(self, model_path: Path) -> None:
        log.info(f"Saving model to {model_path}")
        torch.save(self, model_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            # torch.save is similar to pickle.dump but handles tensors too
        log.info(f"Saved model to {model_path}")

    @classmethod
    def load(cls, model_path: Path, device: str = 'cpu') -> "LanguageModel":
        log.info(f"Loading model from {model_path}")
        model = torch.load(model_path, map_location=device)
            # torch.load is similar to pickle.load but handles tensors too
            # map_location allows loading tensors on different device than saved
        if not isinstance(model, cls):
            raise ValueError(f"Type Error: expected object of type {cls} but got {type(model)} from file {model_path}")
        log.info(f"Loaded model from {model_path}")
        return model

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count   = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")

    def sample_sentence(self, max_length: int = 20) -> List[Wordtype]:
        """Sample a sentence from this language model."""

        sentence = []
        x, y = BOS, BOS  #start with bos
        
        for _ in range(max_length):
            #get probabilities for all possible next words
            probs = []
            for z in self.vocab:
                prob = math.exp(self.log_prob(x, y, z))  #convert log prob to prob for sampling
                probs.append(prob)
            
            #convert to PyTorch tensor for sampling
            probs_tensor = torch.tensor(probs, dtype=torch.float)
            
            #sample next word
            sampled_idx = torch.multinomial(probs_tensor, 1).item()
            z = list(self.vocab)[sampled_idx]  #get the word from vocab
            
            #check if we've reached end of sentence
            if z == EOS:
                break
                
            #add word to sentence and update context
            sentence.append(z)
            x, y = y, z  #shift context
            
        return sentence


##### SPECIFIC FAMILIES OF LANGUAGE MODELS

class CountBasedLanguageModel(LanguageModel):

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )

class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)
        if lambda_ < 0.0:
            raise ValueError(f"Negative lambda argument of {lambda_} could result in negative smoothed probs")
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.


class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # Backoff add-lambda smoothing: recursively backs off to lower-order n-grams
        # Don't forget the difference between the Wordtype z and the
        # 1-element tuple (z,). If you're looking up counts,
        # these will have very different counts!
        
        #recursively get backed off probability z given y
        backoff_prob_zy = self._backoff_prob_bigram(y, z)
        
        return ((self.event_count[x, y, z] + self.lambda_ * self.vocab_size * backoff_prob_zy) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))
    
    def _backoff_prob_bigram(self, y: Wordtype, z: Wordtype) -> float:
        #use formula from slides
        #get backed off probability of z
        backoff_prob_z = self._backoff_prob_unigram(z)
        
        return ((self.event_count[y, z] + self.lambda_ * self.vocab_size * backoff_prob_z) /
                (self.context_count[y,] + self.lambda_ * self.vocab_size))
    
    def _backoff_prob_unigram(self, z: Wordtype) -> float:
        #use formula from slides
        
        uniform_prob = 1.0 / self.vocab_size
        
        return ((self.event_count[z,] + self.lambda_ * self.vocab_size * uniform_prob) /
                (self.context_count[()] + self.lambda_ * self.vocab_size))


class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.
    
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int) -> None:
        super().__init__(vocab)
        if l2 < 0:
            raise ValueError("Negative regularization strength {l2}")
        self.l2: float = l2
        self.epochs: int = epochs

        #read the lexicon of word vectors
        self.embeddings: dict[Wordtype, torch.Tensor] = {}
        
        with open(lexicon_file) as f:
            first_line = f.readline().strip().split()
            vocab_size = int(first_line[0])
            self.dim = int(first_line[1])
            
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:  # word + at least one embedding value
                    word = parts[0]
                    embedding_values = [float(x) for x in parts[1:1+self.dim]]
                    self.embeddings[word] = torch.tensor(embedding_values, dtype=torch.float32)
        
        #create a list of all vocabulary words
        self.vocab_list = list(self.vocab)
        self.word_to_idx = {word: i for i, word in enumerate(self.vocab_list)}

        # We wrap the following matrices in nn.Parameter objects.
        # This lets PyTorch know that these are parameters of the model
        # that should be listed in self.parameters() and will be
        # updated during training.
        #
        # We can also store other tensors in the model class,
        # like constant coefficients that shouldn't be altered by
        # training, but those wouldn't use nn.Parameter.
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)

    def get_embedding(self, word: Wordtype) -> torch.Tensor:
        #Get embedding for a word using OOL for words not in lexicon.
        if word in self.embeddings:
            return self.embeddings[word]
        elif OOL in self.embeddings:
            return self.embeddings[OOL]
        else:
            #if ool is not available, return zero vector
            return torch.zeros(self.dim, dtype=torch.float32)

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        # https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return self.log_prob_tensor(x, y, z).item()

    @typechecked
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        """Return the same value as log_prob, but stored as a tensor."""
        
        # As noted below, it's important to use a tensor for training.
        # Most of your intermediate quantities, like logits below, will
        # also be stored as tensors.  (That is normal in PyTorch, so it
        # would be weird to append `_tensor` to their names.  We only
        # appended `_tensor` to the name of this method to distinguish
        # it from the class's general `log_prob` method.)

        #get logits for all words in vocabulary
        logits = self.logits(x, y)  # shape: (vocab_size,)
        
        #convert to log probabilitiesfor numerical stability
        log_probs = torch.log_softmax(logits, dim=0)  #(vocab_size,)
        
        #get the index of word z in the vocabulary
        if z not in self.word_to_idx:
            #if z is not in vocabulary, it should be OOV

            z = OOV
        
        z_idx = self.word_to_idx[z]
        
        # Return the log probability for word z
        return log_probs[z_idx]

    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor,"vocab"]:
        """Return a vector of the logs of the unnormalized probabilities f(xyz) * θ 
        for the various types z in the vocabulary.
        These are commonly known as "logits" or "log-odds": the values that you 
        exponentiate and renormalize in order to get a probability distribution."""
        
        #get embeddings for context words x and y
        x_emb = self.get_embedding(x)  #(dim,)
        y_emb = self.get_embedding(y)  #(dim,)
        
        #create a tensor of all z embeddings in vocabulary order
        z_embeddings = torch.stack([self.get_embedding(z) for z in self.vocab_list])  #(vocab_size, dim)

        # f(x,y,z) = x^T * X * z + y^T * Y * z
 
        term1 = (x_emb @ self.X) @ z_embeddings.T  #(vocab_size,)
        term2 = (y_emb @ self.Y) @ z_embeddings.T  #(vocab_size,)

        logits = term1 + term2  # shape: (vocab_size,)
        
        return logits

    def train(self, file: Path):    # type: ignore
        
        ### Technically this method shouldn't be called `train`,
        ### because this means it overrides not only `LanguageModel.train` (as desired)
        ### but also `nn.Module.train` (which has a different type). 
        ### However, we won't be trying to use the latter method.
        ### The `type: ignore` comment above tells the type checker to ignore this inconsistency.
        
        # Optimization hyperparameters.
        #use instructions to determine learning rate
        file_str = str(file)
        if 'english' in file_str or 'spanish' in file_str or 'english_spanish' in file_str:
            #language id
            eta0 = 0.01  # 10^-2
            log.info(f"Using learning rate {eta0} for language ID task")
        elif 'gen' in file_str or 'spam' in file_str or 'gen_spam' in file_str:
            #spam detection
            eta0 = 0.00001  # 10^-5
            log.info(f"Using learning rate {eta0} for spam detection task")
        else:
            #default
            eta0 = 0.01
            log.info(f"Using default learning rate {eta0}")

        # This is why we needed the nn.Parameter above.
        # The optimizer needs to know the list of parameters
        # it should be trying to update.
        optimizer = optim.SGD(self.parameters(), lr=eta0)

        # Initialize the parameter matrices to be full of zeros.
        nn.init.zeros_(self.X)   # type: ignore
        nn.init.zeros_(self.Y)   # type: ignore

        N = num_tokens(file)
        log.info("Start optimizing on {N} training tokens...")

        # Training loop for the specified number of epochs
        total_trigrams = N * self.epochs
        trigram_count = 0
        

            
        for epoch in range(self.epochs):
            log.info(f"Starting epoch {epoch + 1}/{self.epochs}")
            
            # Get trigrams for this epoch
            trigrams = list(read_trigrams(file, self.vocab))
            

            trigram_iter = trigrams
            
            for trigram in trigram_iter:
                x, y, z = trigram
                
                #clear gradients from previous iteration
                optimizer.zero_grad()
                
                #Forward pass
                log_prob = self.log_prob_tensor(x, y, z)
                
                nll_loss = -log_prob
                l2_reg = 0.5 * self.l2 * (torch.sum(self.X ** 2) + torch.sum(self.Y ** 2))
            
                objective = nll_loss + l2_reg
                
                #compute gradients
                objective.backward()
                
                #update parameters
                optimizer.step()
                
                trigram_count += 1
                
                if trigram_count % 1000 == 0:
                    log.info(f"{trigram_count}/{total_trigrams} trigrams")
            
            log.info(f"Completed epoch {epoch + 1}/{self.epochs}")
    

        log.info("done optimizing.")

        # So how does the `backward` method work?
        #
        # As Python sees it, your parameters and the values that you compute
        # from them are not actually numbers.  They are `torch.Tensor` objects.
        # A Tensor may represent a numeric scalar, vector, matrix, etc.
        #
        # Every Tensor knows how it was computed.  For example, if you write `a
        # = b + exp(c)`, PyTorch not only computes `a` but also stores
        # backpointers in `a` that remember how the numeric value of `a` depends
        # on the numeric values of `b` and `c`.  In turn, `b` and `c` have their
        # own backpointers that remember what they depend on, and so on, all the
        # way back to the parameters.  This is just like the backpointers in
        # parsing!
        #
        # Every Tensor has a `backward` method that computes the gradient of its
        # numeric value with respect to the parameters, using "back-propagation"
        # through this computation graph.  In particular, once you've computed
        # the forward quantity F_i(θ) as a tensor, you can trace backwards to
        # get its gradient -- i.e., to find out how rapidly it would change if
        # each parameter were changed slightly.

#todo
class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int, 
                 dropout: float = 0.1, batch_size: int = 512, 
                 label_smoothing: float = 0.05,
                 warmup_steps: int = 50, grad_accum_steps: int = 1) -> None:
        super().__init__(vocab, lexicon_file, l2, epochs)
        
        vocab_size = len(self.vocab_list)
        
        #hyperparameters
        self.dropout_rate = dropout
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.warmup_steps = warmup_steps
        self.grad_accum_steps = grad_accum_steps
        
        #dropout for embeddings
        self.dropout = nn.Dropout(p=dropout)
        
        #bias terms
        self.bias_x = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.bias_y = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        
        #unigram indicator features: one weight per vocab word
        self.unigram_weights = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        
        #unigram embedding features: direct z embedding projection
        self.unigram_u = nn.Parameter(torch.zeros(self.dim), requires_grad=True)
        
        #trigram embedding features: low-rank tensor for xyz interaction (reduced rank for speed)
        self.trigram_rank = min(10, self.dim)
        self.T_left = nn.Parameter(torch.zeros((self.dim, self.trigram_rank)), requires_grad=True)
        
        #spelling features: suffixes, prefixes, caps, digits, etc
        self.suffixes = ['-ing', '-ed', '-s', '-ly', '-er', '-est', '-tion', '-ness', '-ment', '-ity']
        self.prefixes = ['un-', 're-', 'in-', 'dis-', 'en-', 'non-', 'pre-', 'mis-']
        n_spell_features = len(self.suffixes) + len(self.prefixes) + 5
        self.spell_weights = nn.Parameter(torch.zeros(n_spell_features), requires_grad=True)
        self._spelling_cache = {}
        
        #repetition features: check if z in recent history
        self.repetition_windows = [2, 5, 10, 20]
        self.repetition_weights = nn.Parameter(torch.zeros(len(self.repetition_windows)), requires_grad=True)
        self.word_history: List[Wordtype] = []
        
        #initialize all parameters
        self._initialize_parameters()
        
        #cache for embeddings
        self.embedding_cache = None
        
    def _initialize_parameters(self):
        """initialize parameters"""
        #xavier uniform for matrices
        nn.init.xavier_uniform_(self.X)
        nn.init.xavier_uniform_(self.Y)
        nn.init.xavier_uniform_(self.T_left)
        
        #small random init for biases and unigram features
        nn.init.uniform_(self.bias_x, -0.01, 0.01)
        nn.init.uniform_(self.bias_y, -0.01, 0.01)
        nn.init.uniform_(self.unigram_weights, -0.01, 0.01)
        nn.init.uniform_(self.unigram_u, -0.01, 0.01)
        nn.init.uniform_(self.spell_weights, -0.01, 0.01)
        nn.init.uniform_(self.repetition_weights, 0.0, 0.1)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """get embeddings for all vocab words"""
        if self.embedding_cache is None:
            z_embeddings = torch.stack([self.get_embedding(z) for z in self.vocab_list])
            self.embedding_cache = z_embeddings
        return self.embedding_cache
    
    def _get_spelling_features(self, word: Wordtype) -> torch.Tensor:
        """extract spelling features for a word"""
        if word in self._spelling_cache:
            return self._spelling_cache[word]
        
        features = []
        word_lower = word.lower()
        
        #suffix features
        for suffix in self.suffixes:
            features.append(1.0 if word_lower.endswith(suffix) else 0.0)
        
        #prefix features  
        for prefix in self.prefixes:
            features.append(1.0 if word_lower.startswith(prefix) else 0.0)
        
        #additional features
        features.append(1.0 if word[0].isupper() else 0.0)
        features.append(1.0 if word.isupper() else 0.0)
        features.append(1.0 if any(c.isdigit() for c in word) else 0.0)
        features.append(1.0 if '-' in word else 0.0)
        features.append(float(len(word)) / 20.0)
        
        feat_tensor = torch.tensor(features, dtype=torch.float32)
        self._spelling_cache[word] = feat_tensor
        return feat_tensor
    
    def _get_repetition_features(self, z: Wordtype, history: List[Wordtype]) -> torch.Tensor:
        """check if word z appeared in recent history"""
        features = []
        for window_size in self.repetition_windows:
            recent = history[-window_size:] if len(history) >= window_size else history
            features.append(1.0 if z in recent else 0.0)
        return torch.tensor(features, dtype=torch.float32)
    
    def logits(self, x: Wordtype, y: Wordtype, history: Optional[List[Wordtype]] = None, use_all_features: bool = False) -> Float[torch.Tensor,"vocab"]:
        """extended log-linear model with multiple feature types
        
        Args:
            use_all_features: if False (default for training), skip expensive spelling features
        """
        
        #get embeddings
        x_emb = self.get_embedding(x)
        y_emb = self.get_embedding(y)
        
        #apply dropout during training
        if self.training:
            x_emb = self.dropout(x_emb)
            y_emb = self.dropout(y_emb)
        
        z_embeddings = self.get_all_embeddings()
        
        #original bigram and skip-bigram features
        term1 = (x_emb @ self.X) @ z_embeddings.T + self.bias_x
        term2 = (y_emb @ self.Y) @ z_embeddings.T + self.bias_y
        logits = term1 + term2
        
        #unigram indicator features
        logits = logits + self.unigram_weights
        
        #unigram embedding features
        logits = logits + (z_embeddings @ self.unigram_u)
        
        #trigram embedding features as low-rank xyz interaction
        x_proj = x_emb @ self.T_left
        y_proj = y_emb @ self.T_left
        xy_combined = x_proj * y_proj
        z_proj = z_embeddings @ self.T_left
        trigram_scores = z_proj @ xy_combined
        logits = logits + trigram_scores
        
        #spelling features
        if use_all_features:
            spell_scores = torch.zeros(len(self.vocab_list))
            for i, z_word in enumerate(self.vocab_list):
                spell_feat = self._get_spelling_features(z_word)
                spell_scores[i] = spell_feat @ self.spell_weights
            logits = logits + spell_scores
        
        #repetition features
        if history is not None and use_all_features:
            rep_scores = torch.zeros(len(self.vocab_list))
            for i, z_word in enumerate(self.vocab_list):
                rep_feat = self._get_repetition_features(z_word, history)
                rep_scores[i] = rep_feat @ self.repetition_weights
            logits = logits + rep_scores
        
        return logits
    
    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """compute log probability with extended features"""
        #update history for repetition features
        self.word_history.append(y)
        if len(self.word_history) > max(self.repetition_windows):
            self.word_history.pop(0)
        
        #handle OOV words - map to OOV if not in vocabulary
        if z not in self.word_to_idx:
            z = OOV
            # If OOV is also not in vocab, return very low probability
            if z not in self.word_to_idx:
                return -math.inf
        
        #compute with history
        with torch.no_grad():
            logits = self.logits(x, y, history=self.word_history)
            log_probs = torch.log_softmax(logits, dim=0)
            z_idx = self.word_to_idx[z]
            return log_probs[z_idx].item()
    
    def label_smoothing_loss(self, logits: torch.Tensor, target_idx: int) -> torch.Tensor:
        """compute label smoothing loss"""
        vocab_size = len(self.vocab_list)
        
        #compute log probabilities
        log_probs = torch.log_softmax(logits, dim=0)
        
        #create smoothed target distribution
        confidence = 1.0 - self.label_smoothing
        smooth_value = self.label_smoothing / (vocab_size - 1)
        
        #true target gets (1-smoothing), others get smoothing/(V-1)
        target_log_prob = log_probs[target_idx]
        smooth_loss = -torch.sum(log_probs) * smooth_value
        
        #combine
        loss = -confidence * target_log_prob + smooth_loss
        
        return loss
    
    def get_lr_schedule(self, step: int, max_steps: int) -> float:
        """learning rate schedule with warmup and cosine annealing"""
        if step < self.warmup_steps:
            #linear warmup
            return step / self.warmup_steps
        else:
            #cosine annealing
            progress = (step - self.warmup_steps) / (max_steps - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    def train(self, file: Path, max_tokens: Optional[int] = None):  
        """training with extended features"""
        
        #adamw optimizer
        base_lr = 0.01
        optimizer = optim.AdamW(self.parameters(), lr=base_lr, 
                               betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=self.l2)
        
        #set model to training mode (PyTorch)
        self.training = True  # Set PyTorch training flag (used by dropout, etc.)
        self.train_mode = True
        
        N = num_tokens(file)
        log.info(f"=== Extended Log-Linear Training on {N} tokens ===")
        log.info(f"Config: dropout={self.dropout_rate}, batch_size={self.batch_size}, "
                f"grad_accum={self.grad_accum_steps}, l2={self.l2}")
        log.info(f"Features: unigram indicators+embeddings, trigram embeddings, spelling, repetition")
        
        #load and shuffle trigrams
        import random
        all_trigrams = list(read_trigrams(file, self.vocab))
        
        if max_tokens is not None and max_tokens < len(all_trigrams):
            trigrams = all_trigrams[:max_tokens]
            log.info(f"Limiting training to {max_tokens} tokens (out of {len(all_trigrams)} available)")
        else:
            trigrams = all_trigrams
        
        total_steps = (len(trigrams) // (self.batch_size * self.grad_accum_steps)) * self.epochs
        log.info(f"Total steps: {total_steps}, Total tokens: {len(trigrams)}")
        
        global_step = 0
        token_count = 0
        
        for epoch in range(self.epochs):
            log.info(f"\n{'='*60}")
            log.info(f"Epoch {epoch + 1}/{self.epochs}")
            log.info(f"{'='*60}")
            
            epoch_loss = 0.0
            num_trained_tokens = 0
            batch_count = 0
            optimizer.zero_grad()
            
            for i in range(0, len(trigrams), self.batch_size):
                batch_trigrams = trigrams[i:i + self.batch_size]
                batch_loss = 0.0
                batch_trained = 0
                
                #process each trigram in batch
                for trigram in batch_trigrams:
                    x, y, z = trigram
                    
                    token_count += 1
                    
                    #print progress every 100 tokens
                    if token_count % 100 == 0:
                        msg = f"Tokens seen: {token_count}/{len(trigrams) * self.epochs}"
                        log.info(msg)
                        print(msg)  # Also print to stdout for visibility
                    
                    #skip if z is not in vocabulary
                    if z not in self.word_to_idx:
                        continue
                    
                    #forward pass
                    logits = self.logits(x, y, history=None)
                    z_idx = self.word_to_idx[z]
                    
                    #compute loss (with label smoothing if configured)
                    if self.label_smoothing > 0:
                        loss = self.label_smoothing_loss(logits, z_idx)
                    else:
                        log_probs = torch.log_softmax(logits, dim=0)
                        loss = -log_probs[z_idx]
                    
                    #backward pass
                    loss.backward()
                    
                    batch_loss += loss.item()
                    batch_trained += 1
                    num_trained_tokens += 1
                
                #update after accumulating gradients
                if (batch_count + 1) % self.grad_accum_steps == 0:
                    #apply learning rate schedule if using warmup
                    if self.warmup_steps > 0:
                        lr_mult = self.get_lr_schedule(global_step, total_steps)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = base_lr * lr_mult
                    
                    #gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    
                    #update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                epoch_loss += batch_loss
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / num_trained_tokens if num_trained_tokens > 0 else 0.0
            log.info(f"\nEpoch {epoch + 1} complete: avg_loss={avg_epoch_loss:.4f} (trained on {num_trained_tokens} tokens)")
        
        #clear cache after training
        self.embedding_cache = None
        
        #set back to evaluation mode
        self.training = False  # Set PyTorch eval flag (used by dropout, etc.)
        self.train_mode = False
        
        log.info(f"\n{'='*60}")
        log.info(f"Training complete! Final loss: {avg_epoch_loss:.4f}")
        log.info(f"{'='*60}")
