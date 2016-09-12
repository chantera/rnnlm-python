#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import rnnlm


class LM:

    def __init__(self, rnnlm_file):
        model = rnnlm.CRnnLME()
        model.setRnnLMFile(rnnlm_file)
        model.setDebugMode(0)
        self.model = model

    def ppl(self, words):
        return 10 ** (- self.model.getProb(rnnlm.StringVector(words)) / len(words))


def main(argv):
    model = LM(rnnlm_file=argv[1])
    print(model.ppl([
        'good',
        'morning'
    ]))
    print(model.ppl([
        'this',
        'is',
        'a',
        'pen',
    ]))


if __name__ == "__main__":
    main(sys.argv)
