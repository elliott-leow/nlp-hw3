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
        
        #eval mode
        if isinstance(model, nn.Module):
            model.eval()
        
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

class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        
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
    '''Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.'''
    
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

        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        
        #cache embeddings
        self._z_embeddings_cache: Optional[torch.Tensor] = None

    def get_embedding(self, word: Wordtype) -> torch.Tensor:
        #Get embedding for a word using OOL for words not in lexicon.
        if word in self.embeddings:
            return self.embeddings[word]
        elif OOL in self.embeddings:
            return self.embeddings[OOL]
        else:
            #if ool is not available, return zero vector
            return torch.zeros(self.dim, dtype=torch.float32)
    
    def get_z_embeddings(self) -> torch.Tensor:

        # check training mode
        if self.training:
            #always rebuild during training to maintain gradient graph
            return torch.stack([self.get_embedding(z) for z in self.vocab_list])
        else:
            #use cache during evaluation for speed
            if self._z_embeddings_cache is None:
                self._z_embeddings_cache = torch.stack([self.get_embedding(z) for z in self.vocab_list])
            return self._z_embeddings_cache

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return self.log_prob_tensor(x, y, z).item()

    @typechecked
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        

        #get logits for all words in vocabulary
        logits = self.logits(x, y)  # shape: (vocab_size,)
        
        #convert to log probabilitiesfor numerical stability
        log_probs = torch.log_softmax(logits, dim=0)  #(vocab_size,)
        
        #get the index of word z in the vocabulary
        if z not in self.word_to_idx:
            #if z is not in vocabulary, check for OOV or OOL
            if OOV in self.word_to_idx:
                z = OOV
            elif OOL in self.word_to_idx:
                z = OOL
            else:
                #low probability if neither OOV nor OOL exists
                return torch.tensor(-1000.0)
        
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
        
        #get cached z embeddings (much faster than rebuilding every time)
        z_embeddings = self.get_z_embeddings()  #(vocab_size, dim)

        # f(x,y,z) = x^T * X * z + y^T * Y * z
 
        term1 = (x_emb @ self.X) @ z_embeddings.T  #(vocab_size,)
        term2 = (y_emb @ self.Y) @ z_embeddings.T  #(vocab_size,)

        logits = term1 + term2  # shape: (vocab_size,)
        
        return logits

    def train(self, file: Path):    # type: ignore
        
        nn.Module.train(self, mode=True)
        
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

        optimizer = optim.SGD(self.parameters(), lr=eta0)

        #initialize the parameter matrices to be full of zeros.
        nn.init.zeros_(self.X)   
        nn.init.zeros_(self.Y)  

        N = num_tokens(file)
        log.info("Start optimizing on {N} training tokens...")

        #training loop for the specified number of epochs
        total_trigrams = N * self.epochs
        trigram_count = 0
        

            
        for epoch in range(self.epochs):
            log.info(f"Starting epoch {epoch + 1}/{self.epochs}")
            
            #get trigrams for this epoch
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
                
                if trigram_count % 100 == 0:
                    msg = f"Tokens seen: {trigram_count}/{total_trigrams}"
                    log.info(msg)
                    print(msg)
            
            log.info(f"Completed epoch {epoch + 1}/{self.epochs}")
    

        log.info("done optimizing.")


class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int) -> None:
        super().__init__(vocab, lexicon_file, l2, epochs)
        
        vocab_size = len(self.vocab_list)
        
        #global bias term (replaces redundant bias_x + bias_y)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        
        #unigram indicator features for output word
        self.unigram_weights = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        
        #word-specific context biases (vocab_size -> 1 mapping for each context word)
        self.x_word_bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.y_word_bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        
        #unigram embedding projection for output word
        self.U = nn.Parameter(torch.zeros(self.dim), requires_grad=True)
        
        #bigram context interaction (x and y interact before combining with z)
        self.context_rank = min(8, self.dim)
        self.C_x = nn.Parameter(torch.zeros((self.dim, self.context_rank)), requires_grad=True)
        self.C_y = nn.Parameter(torch.zeros((self.dim, self.context_rank)), requires_grad=True)
        self.C_out = nn.Parameter(torch.zeros((self.context_rank, self.dim)), requires_grad=True)
        
        #skip-gram feature (x directly with z, skipping y)
        self.Z = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        
        #initialize parameters
        nn.init.uniform_(self.bias, -0.01, 0.01)
        nn.init.uniform_(self.unigram_weights, -0.01, 0.01)
        nn.init.uniform_(self.x_word_bias, -0.01, 0.01)
        nn.init.uniform_(self.y_word_bias, -0.01, 0.01)
        nn.init.uniform_(self.U, -0.01, 0.01)
        nn.init.xavier_uniform_(self.C_x)
        nn.init.xavier_uniform_(self.C_y)
        nn.init.xavier_uniform_(self.C_out)
        nn.init.xavier_uniform_(self.Z)
        
    def logits(self, x: Wordtype, y: Wordtype) -> Float[torch.Tensor,"vocab"]:
        #compute logits with additional features
        
        #get embeddings
        x_emb = self.get_embedding(x)
        y_emb = self.get_embedding(y)
        z_embeddings = self.get_z_embeddings()  # Use cached z_embeddings
        
        #base bigram features: x^T X z + y^T Y z
        term1 = (x_emb @ self.X) @ z_embeddings.T
        term2 = (y_emb @ self.Y) @ z_embeddings.T
        logits = term1 + term2
        
        #global bias
        logits = logits + self.bias
        
        #unigram indicator features
        logits = logits + self.unigram_weights
        
        #unigram embedding features: z^T U
        logits = logits + (z_embeddings @ self.U)
        
        #word-specific context biases
        x_idx = self.word_to_idx.get(x, self.word_to_idx.get(OOV, 0))
        y_idx = self.word_to_idx.get(y, self.word_to_idx.get(OOV, 0))
        logits = logits + self.x_word_bias[x_idx] + self.y_word_bias[y_idx]
        
        #bigram context interaction: (x^T C_x) * (y^T C_y) combined with z
        x_context = x_emb @ self.C_x  #(context_rank,)
        y_context = y_emb @ self.C_y  #(context_rank,)
        xy_interaction = x_context * y_context  #(context_rank,)
        context_feature = xy_interaction @ self.C_out  #(dim,)
        logits = logits + (z_embeddings @ context_feature)
        
        #skip-gram feature: x^T Z z
        skip_gram = (x_emb @ self.Z) @ z_embeddings.T
        logits = logits + skip_gram
        
        return logits
    
    def train(self, file: Path):
        #use same training structure as base model but with adamw
        
        nn.Module.train(self, mode=True)
        
        #determine learning rate based on file name (same logic as base model)
        file_str = str(file)
        if 'english' in file_str or 'spanish' in file_str or 'english_spanish' in file_str:
            #language id
            eta0 = 0.01  # 10^-2
            log.info(f"Using learning rate {eta0} for language ID task")
        elif 'gen' in file_str or 'spam' in file_str or 'gen_spam' in file_str:
            #spam detection
            eta0 = 0.0001  # 10^-5
            log.info(f"Using learning rate {eta0} for spam detection task")
        else:
            #default
            eta0 = 0.01
            log.info(f"Using default learning rate {eta0}")
        
        #adamw optimizer instead of sgd
        optimizer = optim.AdamW(self.parameters(), lr=eta0, weight_decay=self.l2)
        
        #initialize base matrices to zeros
        nn.init.zeros_(self.X)
        nn.init.zeros_(self.Y)
        
        N = num_tokens(file)
        log.info("Start optimizing on {N} training tokens...")
        
        #training loop
        total_trigrams = N * self.epochs
        trigram_count = 0
        
        for epoch in range(self.epochs):
            log.info(f"Starting epoch {epoch + 1}/{self.epochs}")
            
            trigrams = list(read_trigrams(file, self.vocab))
            
            for trigram in trigrams:
                x, y, z = trigram
                
                #clear gradients
                optimizer.zero_grad()
                
                #forward pass
                log_prob = self.log_prob_tensor(x, y, z)
                
                #loss
                nll_loss = -log_prob
                
                #backward pass
                nll_loss.backward()
                
                #update parameters
                optimizer.step()
                
                trigram_count += 1
                
                if trigram_count % 100 == 0:
                    msg = f"Tokens seen: {trigram_count}/{total_trigrams}"
                    log.info(msg)
                    print(msg)
            
            log.info(f"Completed epoch {epoch + 1}/{self.epochs}")
        
        log.info("done optimizing.")
