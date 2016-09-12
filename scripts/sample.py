#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import rnnlm


def main(argv):
    rnnlm_file = argv[1]
    model = rnnlm.CRnnLME()
    model.setRnnLMFile(rnnlm_file)
    model.setDebugMode(1)
    words = [
        'this',
        'is',
        'test',
    ]
    model.getProb(rnnlm.StringVector(words))


if __name__ == "__main__":
    main(sys.argv)
