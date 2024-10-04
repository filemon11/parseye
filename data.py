from supar.utils.tokenizer import Tokenizer
from supar.utils.transform import Transform, Sentence
from supar.utils import Field

import torch

import os
import csv

from typing import Self, Literal, Iterable, Union, Any, Optional, Generator, Iterator


class EyeTracking(Transform):
    r"""
    A :class:`EyeTracking` object factorize eye tracking data into five fields,
    each associated with one or more :class:`~supar.utils.field.Field` objects.

    Attributes:
        five attributes
    """

    fields = ['WORD', 'NFIX', 'FFD', 'GPT', 'TRT', 'FIXPROP']

    def __init__(
        self,
        WORD: Optional[Union[Field, Iterable[Field]]] = None,
        NFIX: Optional[Union[Field, Iterable[Field]]] = None,
        FFD: Optional[Union[Field, Iterable[Field]]] = None,
        GPT: Optional[Union[Field, Iterable[Field]]] = None,
        TRT: Optional[Union[Field, Iterable[Field]]] = None,
        FIXPROP: Optional[Union[Field, Iterable[Field]]] = None
    ) -> Self:
        super().__init__()

        self.WORD = WORD
        self.NFIX = NFIX
        self.FFD = FFD
        self.GPT = GPT
        self.TRT = TRT
        self.FIXPROP = FIXPROP

    @property
    def src(self):
        return self.WORD, self.NFIX, self.FFD, self.GPT, self.TRT  # [self.WORD]

    @property
    def tgt(self):
        return (self.FIXPROP, ) #self.NFIX, self.FFD, self.GPT, self.TRT, self.FIXPROP

    def load(
        self,
        data: Union[str, Iterable],
        lang: Optional[str] = None,
        **kwargs
    ) -> Iterator["EyeTrackingSentence"]:
        r"""
        Args:
            data (Union[str, Iterable]):
                A filename or a list of instances.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.

        Returns:
            A list of :class:`TreeSentence` instances.
        """

        if lang is not None:
            tokenizer = Tokenizer(lang)

        # TODO: change this loading mechanism
        assert isinstance(data, str) and os.path.exists(data)

        index = 0
        if data.endswith('.txt'):
            data = (s.split() if lang is None else tokenizer(s) for s in open(data) if len(s) > 1)
            
            for s in data:
                yield EyeTrackingSentence(self, s, index)
                index += 1
        
        elif data.endswith('.csv'):
            index = None

            data = csv.reader(open(data, "r"))

            sentence = []
            nFix_sen = []
            ffd_sen = []
            gpt_sen = []
            trt_sen = []
            fixProp_sen = []
            
            for sentence_id, word_id, word, nFix, ffd, gpt, trt, fixProp in data:
                if sentence_id == "sentence_id":
                    continue
                
                sentence_id = int(sentence_id)
                nFix = float(nFix)
                ffd = float(ffd)
                gpt = float(gpt)
                trt = float(trt)
                fixProp = float(fixProp)

                if word.endswith("<EOS>"):
                    word = word[:-5]


                if index is None:
                    index = sentence_id

                if sentence_id  > index:
                    yield EyeTrackingSentence(self, sentence, index,
                                              nFix_sen = nFix_sen,
                                              ffd_sen = ffd_sen,
                                              gpt_sen = gpt_sen,
                                              trt_sen = trt_sen,
                                              fixProp_sen = fixProp_sen)
                    index += 1
                    sentence = [word]
                    nFix_sen = [nFix]
                    ffd_sen = [ffd]
                    gpt_sen = [gpt]
                    trt_sen = [trt]
                    fixProp_sen = [fixProp]
                
                else:
                    sentence.append(word)
                    nFix_sen.append(nFix)
                    ffd_sen.append(ffd)
                    gpt_sen.append(gpt)
                    trt_sen.append(trt)
                    fixProp_sen.append(fixProp)
        
        else:
            raise Exception("Unknown data format.")


class EyeTrackingSentence(Sentence):
    r"""
    Args:
        transform (AttachJuxtaposeTree):
            A :class:`AttachJuxtaposeTree` object.
        tree (nltk.tree.Tree):
            A :class:`nltk.tree.Tree` object.
        index (Optional[int]):
            Index of the sentence in the corpus. Default: ``None``.
    """

    def __init__(
        self,
        transform: EyeTracking,
        sentence: list[str],
        index: Optional[int] = None,
        **kwargs
    ):
        super().__init__(transform, index)

        self.values = [sentence]
        self.values.extend(kwargs.values())

        length = len(sentence)
        assert all(len(v) == length for v in self.values)

        #print(self.values)

    def __repr__(self):
        return f"Sentence({self.values})"

    def pretty_print(self):
        return f"Sentence({self.values})"

class FloatField(Field):
    def transform(self, sequences: Iterable[list[float]]) -> Iterable[torch.Tensor]:
        r"""
        Turns a list of sequences that use this field into tensors.

        Each sequence is first preprocessed and then numericalized if needed.

        Args:
            sequences (Iterable[List[str]]):
                A list of sequences.

        Returns:
            A list of tensors transformed from the input sequences.
        """

        for seq in sequences:
            #seq = self.preprocess(seq)
            
            #if self.bos:
            #    seq = [self.bos_index] + seq
            #if self.delay > 0:
            #    seq = seq + [self.pad_index for _ in range(self.delay)]
            #if self.eos:
            #    seq = seq + [self.eos_index]

            yield torch.tensor(seq, dtype=torch.float32)
